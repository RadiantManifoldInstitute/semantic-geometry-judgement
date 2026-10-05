#!/usr/bin/ruby
# frozen_string_literal: true

# Independent Path B recomputation of the executable oracle, proofs, and sealed joins.

require 'digest'
require 'json'
require 'optparse'
require 'fileutils'

def normalized(value)
  case value
  when String then value.unicode_normalize(:nfc)
  when Array then value.map { |item| normalized(item) }
  when Hash then value.keys.sort_by(&:b).to_h { |key| [key.unicode_normalize(:nfc), normalized(value.fetch(key))] }
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
policies = read_jsonl(File.join(primitive, 'policy-catalog.jsonl'))
actions = read_jsonl(File.join(primitive, 'action-cases.jsonl'))
pairs = read_jsonl(File.join(primitive, 'pair-ir.jsonl'))
rendered = read_jsonl(File.join(primitive, 'rendered-pairs.jsonl')).to_h { |row| [row.fetch('pair_id'), row] }
salt = Digest::SHA256.hexdigest("SYSTEM-K-K-F1E|41385001|JOIN|V0.1\n")

policy_oracle = policies.map do |policy|
  sealed({
    'schema_version' => 'K_F1E_EXECUTABLE_POLICY_ORACLE_V0_1', 'policy_id' => policy.fetch('policy_id'),
    'family_id' => policy.fetch('family_id'), 'slot' => policy.fetch('slot'),
    'predicate' => {'operator' => 'SLOT_MEMBERSHIP_AND_RELATION_LOOKUP', 'action_field' => 'relations_by_slot', 'key' => policy.fetch('slot')},
    'output_domain' => %w[NONE ALIGNED OPPOSED AMBIGUOUS]
  })
end
action_oracle = actions.map { |row| sealed(row.reject { |key, _| key == 'action_text' }) }
proofs = []
calibration_gold = []
sealed_gold = []
calibration_join = []
sealed_join = []

pairs.each do |pair|
  surface = rendered.fetch(pair.fetch('pair_id'))
  combined = [surface.fetch('action_text'), surface.fetch('policy_text'), surface.fetch('aligned_reference_text'), surface.fetch('opposed_reference_text')].join("\n") + "\n"
  proof = sealed({
    'schema_version' => 'K_F1E_SURFACE_PROOF_V0_1', 'pair_id' => pair.fetch('pair_id'), 'action_id' => pair.fetch('action_id'),
    'policy_id' => pair.fetch('policy_id'), 'surface_class' => pair.fetch('surface_class'), 'pair_ir_sha256' => digest_record(pair),
    'rendered_text_sha256' => Digest::SHA256.hexdigest(combined), 'semantics_preserved_bit' => 1
  })
  proofs << proof
  gold_base = pair.merge({
    'schema_version' => 'K_F1E_PAIR_GOLD_V0_1',
    'action_token_count' => surface.fetch('action_text').scan(/[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*/).length,
    'policy_token_count' => surface.fetch('policy_text').scan(/[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*/).length,
    'surface_proof_payload_sha256' => proof.fetch('payload_sha256')
  })
  gold = sealed(gold_base)
  blind_pair_id = Digest::SHA256.hexdigest("#{salt}|#{pair.fetch('pair_id')}")
  blind_action_id = Digest::SHA256.hexdigest("#{salt}|#{pair.fetch('action_id')}")
  blind_policy_id = Digest::SHA256.hexdigest("#{salt}|#{pair.fetch('policy_id')}|#{pair.fetch('abstraction')}")
  join = sealed({
    'schema_version' => 'K_F1E_JOIN_MAP_V0_1', 'blind_pair_id' => blind_pair_id, 'blind_action_id' => blind_action_id,
    'blind_policy_id' => blind_policy_id, 'pair_id' => pair.fetch('pair_id'), 'action_id' => pair.fetch('action_id'),
    'policy_id' => pair.fetch('policy_id'), 'gold_payload_sha256' => gold.fetch('payload_sha256'), 'join_salt_version_sha256' => salt
  })
  if pair.fetch('split') == 'CALIBRATION'
    calibration_gold << gold
    calibration_join << join
  else
    sealed_gold << gold
    sealed_join << join
  end
end

counts = contract.fetch('counts')
observed = {'actions' => actions.length, 'pairs' => pairs.length, 'calibration_pairs' => calibration_gold.length, 'sealed_pairs' => sealed_gold.length}
raise "census mismatch #{observed}" unless observed.all? { |key, value| counts.fetch(key) == value }

output = options.fetch(:records)
{
  'policy-oracle.jsonl' => policy_oracle, 'action-oracle.jsonl' => action_oracle,
  'surface-inheritance-proof.jsonl' => proofs, 'calibration-gold.jsonl' => calibration_gold,
  'sealed-gold.jsonl' => sealed_gold, 'calibration-join.jsonl' => calibration_join, 'sealed-join.jsonl' => sealed_join
}.each { |filename, rows| write_jsonl(File.join(output, filename), rows) }
