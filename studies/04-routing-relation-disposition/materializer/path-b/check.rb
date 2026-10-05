#!/usr/bin/env ruby
# frozen_string_literal: true

require 'digest'
require 'fileutils'
require 'json'
require 'optparse'

def normalized(value)
  case value
  when String then value.unicode_normalize(:nfc)
  when Array then value.map { |item| normalized(item) }
  when Hash then value.keys.sort_by(&:b).to_h { |key| [key.unicode_normalize(:nfc), normalized(value.fetch(key))] }
  else value
  end
end

def canonical(value)
  JSON.generate(normalized(value)) + "\n"
end

def digest(value)
  Digest::SHA256.hexdigest(canonical(value))
end

def sealed(value)
  value.merge('payload_sha256' => digest(value))
end

def read_jsonl(path)
  File.readlines(path, chomp: true).reject(&:empty?).map { |line| JSON.parse(line) }
end

def write_jsonl(path, rows)
  FileUtils.mkdir_p(File.dirname(path))
  File.open(path, 'wb') { |file| rows.each { |row| file.write(canonical(row)) } }
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
policy_by_id = policies.to_h { |row| [row.fetch('policy_id'), row] }
action_by_id = actions.to_h { |row| [row.fetch('action_id'), row] }
salt = Digest::SHA256.hexdigest("SYSTEM-K-K-F1F|80412026|JOIN|V0.1\n")

policy_oracle = policies.map do |policy|
  sealed({
    'schema_version' => 'K_F1F_EXECUTABLE_POLICY_ORACLE_V0_1', 'policy_id' => policy.fetch('policy_id'),
    'family_id' => policy.fetch('family_id'), 'predicate' => 'GENERATED_GOVERNING_MEMBERSHIP_TABLE_LOOKUP',
    'output_domain' => %w[ALIGNED OPPOSED AMBIGUOUS NON_GOVERNING]
  })
end
action_oracle = actions.map { |row| sealed(row.reject { |key, _| key == 'text' }) }
proofs = []
gold = {'CALIBRATION' => [], 'VALIDATION' => [], 'SEALED' => []}
joins = {'CALIBRATION' => [], 'VALIDATION' => [], 'SEALED' => []}
per_family = Hash.new { |hash, key| hash[key] = Hash.new(0) }

pairs.each do |pair|
  action = action_by_id.fetch(pair.fetch('action_id'))
  policy = policy_by_id.fetch(pair.fetch('policy_id'))
  proof = sealed({
    'schema_version' => 'K_F1F_PERTURBATION_PROOF_V0_1', 'row_id' => pair.fetch('row_id'),
    'perturbation_id' => action.fetch('action_id'), 'base_id' => pair.fetch('base_action_id'), 'deterministic_seed' => 17_022,
    'base_action_id' => pair.fetch('base_action_id'), 'perturbation_kind' => pair.fetch('perturbation_kind'),
    'semantics_preserved' => pair.fetch('perturbation_kind') != 'EXPLICIT_NEGATION',
    'action_text_sha256' => Digest::SHA256.hexdigest(action.fetch('text').unicode_normalize(:nfc)), 'pair_ir_sha256' => digest(pair)
  })
  proofs << proof
  pair_gold = sealed(pair.merge('schema_version' => 'K_F1F_PAIR_GOLD_V0_1', 'perturbation_proof_payload_sha256' => proof.fetch('payload_sha256')))
  blind_pair_id = Digest::SHA256.hexdigest("#{salt}|#{pair.fetch('row_id')}")
  join = sealed({
    'schema_version' => 'K_F1F_JOIN_MAP_V0_1', 'blind_pair_id' => blind_pair_id,
    'blind_action_id' => Digest::SHA256.hexdigest("#{salt}|#{pair.fetch('action_id')}"),
    'blind_policy_id' => Digest::SHA256.hexdigest("#{salt}|#{pair.fetch('policy_id')}"),
    'row_id' => pair.fetch('row_id'), 'gold_payload_sha256' => pair_gold.fetch('payload_sha256'),
    'join_salt_sha256' => salt
  })
  gold.fetch(pair.fetch('split')) << pair_gold
  joins.fetch(pair.fetch('split')) << join
  next unless pair.fetch('perturbation_kind') == 'BASE'

  counts = per_family[pair.fetch('family_id')]
  counts["pairs_#{pair.fetch('policy_set_size')}"] += 1
  counts["relation_#{pair.fetch('relation')}"] += 1
  counts["governing_cardinality_#{pair.fetch('cardinality')}"] += pair.fetch('governing')
end

actions.each do |action|
  next unless action.fetch('perturbation_kind') == 'BASE'

  counts = per_family[action.fetch('family_id')]
  counts["actions_#{action.fetch('policy_set_size')}"] += 1
  counts["cardinality_#{action.fetch('cardinality')}_actions"] += 1
  counts["disposition_#{action.fetch('expected_disposition')}"] += 1
end

observed = {
  'actions' => actions.length, 'pairs' => pairs.length,
  'calibration_actions' => actions.count { |row| row.fetch('split') == 'CALIBRATION' },
  'validation_actions' => actions.count { |row| row.fetch('split') == 'VALIDATION' },
  'sealed_actions' => actions.count { |row| row.fetch('split') == 'SEALED' },
  'calibration_pairs' => gold.fetch('CALIBRATION').length,
  'validation_pairs' => gold.fetch('VALIDATION').length,
  'sealed_pairs' => gold.fetch('SEALED').length,
  'base_actions' => actions.count { |row| row.fetch('perturbation_kind') == 'BASE' },
  'base_pairs' => pairs.count { |row| row.fetch('perturbation_kind') == 'BASE' }
}
raise "CENSUS_MISMATCH:#{observed}" unless observed == contract.fetch('counts')

denominators = {
  'schema_version' => 'K_F1F_DENOMINATOR_MANIFEST_V0_1', 'counts' => observed,
  'per_family_base' => per_family.keys.sort.to_h { |family| [family, per_family.fetch(family).keys.sort.to_h { |key| [key, per_family.fetch(family).fetch(key)] }] },
  'base_expected' => contract.fetch('base_per_family'), 'perturbation_kinds' => contract.fetch('perturbation_kinds')
}

output = options.fetch(:records)
policy_oracle.sort_by! { |row| [row.fetch('family_id'), row.fetch('policy_id')] }
action_oracle.sort_by! { |row| [row.fetch('family_id'), row.fetch('action_id')] }
proofs.sort_by! { |row| row.fetch('row_id') }
gold.each_value { |rows| rows.sort_by! { |row| [row.fetch('family_id'), row.fetch('action_id'), row.fetch('policy_id')] } }
joins.each_value { |rows| rows.sort_by! { |row| row.fetch('row_id') } }
write_jsonl(File.join(output, 'policy-oracle.jsonl'), policy_oracle)
write_jsonl(File.join(output, 'action-oracle.jsonl'), action_oracle)
write_jsonl(File.join(output, 'perturbation-proofs.jsonl'), proofs)
%w[CALIBRATION VALIDATION SEALED].each do |split|
  write_jsonl(File.join(output, "#{split.downcase}-gold.jsonl"), gold.fetch(split))
  write_jsonl(File.join(output, "#{split.downcase}-join.jsonl"), joins.fetch(split))
end
File.binwrite(File.join(output, 'denominator-manifest.json'), canonical(denominators))
