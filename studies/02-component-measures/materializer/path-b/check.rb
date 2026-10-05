#!/usr/bin/ruby
# frozen_string_literal: true

# Independent Path B recomputation of K-F1D gold, proofs, and sealed joins.

require 'digest'
require 'json'
require 'optparse'
require 'fileutils'

def normalized(value)
  case value
  when String then value.unicode_normalize(:nfc)
  when Array then value.map { |item| normalized(item) }
  when Hash
    value.keys.sort_by(&:b).to_h { |key| [key.unicode_normalize(:nfc), normalized(value.fetch(key))] }
  else value
  end
end

def canonical(value)
  JSON.generate(normalized(value))
end

def digest_record(fields)
  Digest::SHA256.hexdigest(canonical(fields) + "\n")
end

def sealed(fields)
  raise 'payload_sha256 supplied by caller' if fields.key?('payload_sha256')

  fields.merge('payload_sha256' => digest_record(fields))
end

def read_jsonl(path)
  File.readlines(path, chomp: true).reject(&:empty?).map { |line| JSON.parse(line) }
end

def write_jsonl(path, rows)
  FileUtils.mkdir_p(File.dirname(path))
  File.open(path, 'wb') { |file| rows.each { |row| file.write(canonical(row) + "\n") } }
end

options = {}
OptionParser.new do |parser|
  parser.on('--contract PATH') { |value| options[:contract] = value }
  parser.on('--primitive-dir PATH') { |value| options[:primitive] = value }
  parser.on('--record-dir PATH') { |value| options[:records] = value }
end.parse!
abort 'required arguments missing' unless options.values_at(:contract, :primitive, :records).all?

contract = JSON.parse(File.read(options.fetch(:contract), encoding: 'UTF-8'))
primitive = options.fetch(:primitive)
pairs = read_jsonl(File.join(primitive, 'pair-ir.jsonl'))
surfaces_by_pair = read_jsonl(File.join(primitive, 'rendered-surfaces.jsonl')).group_by { |row| row.fetch('pair_id') }
join_salt = Digest::SHA256.hexdigest("SYSTEM-K-K-F1D|41384001|JOIN|V0.1\n")

canonical_gold = []
surface_proofs = []
calibration_gold = []
sealed_gold = []
calibration_join = []
sealed_join = []

pairs.each do |pair|
  pair_ir_sha = digest_record(pair)
  base = pair.reject { |key, _value| key == 'scenario_phrase' }
  base['schema_version'] = 'K_F1D_CANONICAL_GOLD_V0_1'
  base['pair_ir_sha256'] = pair_ir_sha
  canonical_row = sealed(base)
  canonical_gold << canonical_row
  surfaces_by_pair.fetch(pair.fetch('pair_id')).each do |surface|
    action = surface.fetch('action_text')
    reference = surface.fetch('reference_text')
    rendered_sha = Digest::SHA256.hexdigest(action + "\n" + reference + "\n")
    proof = sealed({
      'schema_version' => 'K_F1D_SURFACE_PROOF_V0_1',
      'surface_id' => surface.fetch('surface_id'),
      'pair_id' => pair.fetch('pair_id'),
      'surface_class' => surface.fetch('surface_class'),
      'canonical_gold_payload_sha256' => canonical_row.fetch('payload_sha256'),
      'rendered_text_sha256' => rendered_sha,
      'semantics_preserved_bit' => 1
    })
    surface_proofs << proof
    gold_base = pair.reject { |key, _value| %w[schema_version scenario_phrase].include?(key) }
    gold = sealed(gold_base.merge({
      'schema_version' => 'K_F1D_SURFACE_GOLD_V0_1',
      'surface_id' => surface.fetch('surface_id'),
      'surface_class' => surface.fetch('surface_class'),
      'action_token_count' => action.scan(/[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*/).length,
      'reference_token_count' => reference.scan(/[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*/).length,
      'canonical_gold_payload_sha256' => canonical_row.fetch('payload_sha256'),
      'surface_proof_payload_sha256' => proof.fetch('payload_sha256')
    }))
    blind_pair_id = Digest::SHA256.hexdigest("#{join_salt}|#{pair.fetch('pair_id')}")
    blind_surface_id = Digest::SHA256.hexdigest("#{join_salt}|#{surface.fetch('surface_id')}")
    join = sealed({
      'schema_version' => 'K_F1D_JOIN_MAP_V0_1',
      'blind_pair_id' => blind_pair_id,
      'blind_surface_id' => blind_surface_id,
      'pair_id' => pair.fetch('pair_id'),
      'surface_id' => surface.fetch('surface_id'),
      'surface_gold_payload_sha256' => gold.fetch('payload_sha256'),
      'join_salt_version_sha256' => join_salt
    })
    if pair.fetch('split') == 'CALIBRATION'
      calibration_gold << gold
      calibration_join << join
    else
      sealed_gold << gold
      sealed_join << join
    end
  end
end

expected = contract.fetch('counts')
observed = {
  'canonical_pairs' => pairs.length,
  'rendered_surfaces' => surfaces_by_pair.values.sum(&:length),
  'calibration_gold_records' => calibration_gold.length,
  'sealed_gold_records' => sealed_gold.length,
  'calibration_join_records' => calibration_join.length,
  'sealed_join_records' => sealed_join.length
}
raise "census mismatch #{observed}" unless observed.all? { |key, value| expected.fetch(key) == value }

output = options.fetch(:records)
write_jsonl(File.join(output, 'canonical-gold.jsonl'), canonical_gold)
write_jsonl(File.join(output, 'surface-inheritance-proof.jsonl'), surface_proofs)
write_jsonl(File.join(output, 'calibration-gold.jsonl'), calibration_gold)
write_jsonl(File.join(output, 'sealed-gold.jsonl'), sealed_gold)
write_jsonl(File.join(output, 'calibration-join.jsonl'), calibration_join)
write_jsonl(File.join(output, 'sealed-join.jsonl'), sealed_join)
