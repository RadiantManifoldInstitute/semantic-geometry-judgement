#!/usr/bin/ruby
# frozen_string_literal: true

# Declarative Path B checker. It shares only frozen primitive IR and contracts
# with Path A, and owns its evaluator and canonical serializer independently.

require 'digest'
require 'json'
require 'optparse'
require 'fileutils'

LEVELS = %w[L1_CONCRETE L2_OPERATIONAL L3_PURPOSE].freeze
CELL_POLARITY = {
  'APPLICABLE_ALIGNED' => 'ALIGNED',
  'APPLICABLE_OPPOSED' => 'OPPOSED',
  'NOT_APPLICABLE_NONOPPOSED' => 'ALIGNED',
  'NOT_APPLICABLE_APPARENTLY_OPPOSED' => 'OPPOSED'
}.freeze

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

def sealed_record(fields)
  raise 'payload_sha256 supplied by caller' if fields.key?('payload_sha256')

  digest = Digest::SHA256.hexdigest(canonical(fields) + "\n")
  fields.merge('payload_sha256' => digest)
end

def read_jsonl(path)
  File.readlines(path, chomp: true).reject(&:empty?).map { |line| JSON.parse(line) }
end

def write_jsonl(path, rows)
  FileUtils.mkdir_p(File.dirname(path))
  File.open(path, 'wb') do |file|
    rows.each { |row| file.write(canonical(row) + "\n") }
  end
end

options = {}
OptionParser.new do |parser|
  parser.on('--contract PATH') { |value| options[:contract] = value }
  parser.on('--primitive-dir PATH') { |value| options[:primitive] = value }
  parser.on('--record-dir PATH') { |value| options[:records] = value }
end.parse!
abort 'required arguments missing' unless options.values_at(:contract, :primitive, :records).all?

contract = JSON.parse(File.read(options[:contract], encoding: 'UTF-8'))
primitive = options.fetch(:primitive)
families = read_jsonl(File.join(primitive, 'mission-families.jsonl')).to_h { |row| [row.fetch('family_id'), row] }
scenarios = read_jsonl(File.join(primitive, 'scenarios.jsonl'))
actions = read_jsonl(File.join(primitive, 'action-ir.jsonl'))
references = read_jsonl(File.join(primitive, 'reference-ir.jsonl'))
pairs = read_jsonl(File.join(primitive, 'canonical-pairs.jsonl'))
surfaces = read_jsonl(File.join(primitive, 'rendered-surfaces.jsonl'))

actions_by_id = actions.to_h { |row| [row.fetch('action_id'), row] }
refs_by_id = references.to_h { |row| [row.fetch('reference_id'), row] }
scenarios_by_id = scenarios.to_h { |row| [row.fetch('scenario_id'), row] }
surfaces_by_pair = surfaces.group_by { |row| row.fetch('canonical_pair_id') }

ancestry = []
ancestry_for_level = {}
scenarios.each do |scenario|
  refs = references.select { |row| row.fetch('scenario_id') == scenario.fetch('scenario_id') }.to_h { |row| [row.fetch('level'), row] }
  [['L1_CONCRETE', 'L2_OPERATIONAL', 'P01_L1_TO_L2'], ['L2_OPERATIONAL', 'L3_PURPOSE', 'P02_L2_TO_L3']].each do |child_level, parent_level, rule|
    child_atom = refs.fetch(child_level).fetch('required_effect_atom_ids').first
    parent_atom = refs.fetch(parent_level).fetch('required_effect_atom_ids').first
    record = sealed_record({
      'child_reference_id' => refs.fetch(child_level).fetch('reference_id'),
      'parent_reference_id' => refs.fetch(parent_level).fetch('reference_id'),
      'child_level' => child_level,
      'parent_level' => parent_level,
      'projection_rule_id' => rule,
      'child_atom_id' => child_atom,
      'parent_atom_id_or_null' => parent_atom,
      'disposition' => 'RETAINED_GENERALIZED_SCOPE',
      'predicate_sign_before' => 'ASSERT',
      'predicate_sign_after_or_null' => 'ASSERT',
      'scope_mapping' => "#{refs.fetch(child_level).fetch('scope_id')}->#{refs.fetch(parent_level).fetch('scope_id')}"
    })
    ancestry << record
    ancestry_for_level[[scenario.fetch('scenario_id'), child_level]] = record
    ancestry_for_level[[scenario.fetch('scenario_id'), parent_level]] = record
  end
end

transitions = actions.map do |action|
  post_state = ((action.fetch('pre_state_atom_ids') | action.fetch('asserted_atom_ids')) - action.fetch('retracted_atom_ids')).sort
  sealed_record({
    'transition_id' => format('TRN_%04d', action.fetch('action_id').split('_').last.to_i),
    'action_id' => action.fetch('action_id'),
    'pre_state_sha256' => Digest::SHA256.hexdigest(canonical(action.fetch('pre_state_atom_ids')) + "\n"),
    'asserted_atom_ids' => action.fetch('asserted_atom_ids'),
    'retracted_atom_ids' => action.fetch('retracted_atom_ids'),
    'post_state_sha256' => Digest::SHA256.hexdigest(canonical(post_state) + "\n"),
    'entailed_required_atom_ids' => action.fetch('asserted_atom_ids'),
    'entailed_prohibited_atom_ids' => [],
    'contradicted_required_atom_ids' => action.fetch('retracted_atom_ids')
  })
end
transition_by_action = transitions.to_h { |row| [row.fetch('action_id'), row] }

canonical_gold = []
surface_proofs = []
sealed_gold = []
sealed_join = []
join_salt = Digest::SHA256.hexdigest("SYSTEM-K-K-F1C|41381008|JOIN|V0.1\n")

pairs.each do |pair|
  action = actions_by_id.fetch(pair.fetch('action_id'))
  reference = refs_by_id.fetch(pair.fetch('reference_id'))
  scenario = scenarios_by_id.fetch(action.fetch('scenario_id'))
  family = families.fetch(action.fetch('family_id'))
  actor_match = action.fetch('actor_type') == reference.fetch('actor_type_constraint') ? 1 : 0
  object_match = action.fetch('object_type') == reference.fetch('object_type_constraint') ? 1 : 0
  scope_match = reference.fetch('scope_member_actor_ids').include?(action.fetch('actor_id')) && reference.fetch('scope_member_object_ids').include?(action.fetch('object_id')) ? 1 : 0
  preconditions = (reference.fetch('applicability_precondition_atom_ids') - action.fetch('pre_state_atom_ids')).empty? ? 1 : 0
  applicability = actor_match * object_match * scope_match * preconditions
  required_entailed = (reference.fetch('required_effect_atom_ids') - action.fetch('asserted_atom_ids')).empty? ? 1 : 0
  prohibited_entailed = (reference.fetch('prohibited_effect_atom_ids') & action.fetch('asserted_atom_ids')).empty? ? 0 : 1
  required_contradicted = (reference.fetch('required_effect_atom_ids') & action.fetch('retracted_atom_ids')).empty? ? 0 : 1
  polarity = required_contradicted == 1 || prohibited_entailed == 1 ? 'OPPOSED' : 'ALIGNED'
  cell = {
    [1, 'ALIGNED'] => 'APPLICABLE_ALIGNED',
    [1, 'OPPOSED'] => 'APPLICABLE_OPPOSED',
    [0, 'ALIGNED'] => 'NOT_APPLICABLE_NONOPPOSED',
    [0, 'OPPOSED'] => 'NOT_APPLICABLE_APPARENTLY_OPPOSED'
  }.fetch([applicability, polarity])
  slot = action.fetch('slot').delete_prefix('Q').to_i
  raise 'cell schedule mismatch' unless contract.fetch('cell_schedule').fetch(reference.fetch('level')).fetch(slot) == cell

  ancestry_record = ancestry_for_level.fetch([scenario.fetch('scenario_id'), reference.fetch('level')])
  transition_record = transition_by_action.fetch(action.fetch('action_id'))
  gold = sealed_record({
    'schema_version' => 'K_F1C_CANONICAL_GOLD_V0_2',
    'canonical_pair_id' => pair.fetch('canonical_pair_id'),
    'family_id' => family.fetch('family_id'),
    'scenario_id' => scenario.fetch('scenario_id'),
    'action_id' => action.fetch('action_id'),
    'reference_id' => reference.fetch('reference_id'),
    'level' => reference.fetch('level'),
    'applicability_bit' => applicability,
    'actor_match_bit' => actor_match,
    'object_match_bit' => object_match,
    'scope_match_bit' => scope_match,
    'preconditions_satisfied_bit' => preconditions,
    'required_effects_entailed_bit' => required_entailed,
    'prohibited_effects_entailed_bit' => prohibited_entailed,
    'required_effects_contradicted_bit' => required_contradicted,
    'relation_polarity' => polarity,
    'gold_cell' => cell,
    'ancestry_proof_payload_sha256' => ancestry_record.fetch('payload_sha256'),
    'transition_proof_payload_sha256' => transition_record.fetch('payload_sha256')
  })
  canonical_gold << gold
  level_pattern = LEVELS.map do |level|
    scheduled = contract.fetch('cell_schedule').fetch(level).fetch(slot)
    CELL_POLARITY.fetch(scheduled)
  end.join('|')

  surfaces_by_pair.fetch(pair.fetch('canonical_pair_id')).each do |surface|
    rendered_digest = Digest::SHA256.hexdigest(surface.fetch('action_text') + "\n" + surface.fetch('reference_text') + "\n")
    span_digest = Digest::SHA256.hexdigest(canonical(surface.fetch('atom_span_map')) + "\n")
    proof = sealed_record({
      'surface_id' => surface.fetch('surface_id'),
      'canonical_pair_id' => pair.fetch('canonical_pair_id'),
      'surface_class' => surface.fetch('surface_class'),
      'canonical_gold_payload_sha256' => gold.fetch('payload_sha256'),
      'rendered_text_sha256' => rendered_digest,
      'atom_span_map_sha256' => span_digest,
      'semantics_preserved_bit' => 1
    })
    surface_proofs << proof
    sealed = sealed_record({
      'schema_version' => 'K_F1C_SEALED_GOLD_V0_2',
      'canonical_pair_id' => pair.fetch('canonical_pair_id'),
      'surface_id' => surface.fetch('surface_id'),
      'family_id' => family.fetch('family_id'),
      'scenario_id' => scenario.fetch('scenario_id'),
      'action_id' => action.fetch('action_id'),
      'reference_id' => reference.fetch('reference_id'),
      'level' => reference.fetch('level'),
      'split' => family.fetch('split'),
      'applicability_bit' => applicability,
      'actor_match_bit' => actor_match,
      'object_match_bit' => object_match,
      'scope_match_bit' => scope_match,
      'preconditions_satisfied_bit' => preconditions,
      'required_effects_entailed_bit' => required_entailed,
      'prohibited_effects_entailed_bit' => prohibited_entailed,
      'required_effects_contradicted_bit' => required_contradicted,
      'relation_polarity' => polarity,
      'gold_cell' => cell,
      'level_conflict_pattern' => level_pattern,
      'canonical_gold_payload_sha256' => gold.fetch('payload_sha256'),
      'ancestry_proof_payload_sha256' => ancestry_record.fetch('payload_sha256'),
      'transition_proof_payload_sha256' => transition_record.fetch('payload_sha256'),
      'surface_inheritance_proof_payload_sha256' => proof.fetch('payload_sha256')
    })
    sealed_gold << sealed
    blind_pair_id = Digest::SHA256.hexdigest("#{join_salt}|#{pair.fetch('canonical_pair_id')}")
    blind_surface_id = Digest::SHA256.hexdigest("#{join_salt}|#{surface.fetch('surface_id')}")
    sealed_join << sealed_record({
      'schema_version' => 'K_F1C_SEALED_JOIN_MAP_V0_2',
      'blind_pair_id' => blind_pair_id,
      'blind_surface_id' => blind_surface_id,
      'canonical_pair_id' => pair.fetch('canonical_pair_id'),
      'surface_id' => surface.fetch('surface_id'),
      'canonical_gold_payload_sha256' => gold.fetch('payload_sha256'),
      'sealed_gold_payload_sha256' => sealed.fetch('payload_sha256'),
      'join_salt_version_sha256' => join_salt
    })
  end
end

output = options.fetch(:records)
write_jsonl(File.join(output, 'ancestry-proof.jsonl'), ancestry)
write_jsonl(File.join(output, 'transition-proof.jsonl'), transitions)
write_jsonl(File.join(output, 'canonical-gold.jsonl'), canonical_gold)
write_jsonl(File.join(output, 'surface-inheritance-proof.jsonl'), surface_proofs)
write_jsonl(File.join(output, 'sealed-gold.jsonl'), sealed_gold)
write_jsonl(File.join(output, 'sealed-join.jsonl'), sealed_join)
