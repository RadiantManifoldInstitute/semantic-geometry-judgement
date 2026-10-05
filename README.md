# Can Semantic Geometry Teach an AI Judgement?

Scientific companion v0.1.0 for the paper by Thomson D. Nguy.
This version supplies frozen scientific kernels, saved results and offline
recalculation inputs. Full experimental replication is not claimed.

## Methods and Evidence

| Study | Scientific methods/source | Saved replica-a results | Offline inputs |
|---|---|---|---|
| C: scalar and norms | [C](studies/01-scalar-and-norms/) | [C results](results/C/scientific-result.json) | [Gold/joins](gold/C/), [configuration](config/C/) |
| D: component measures | [D](studies/02-component-measures/) | [D results](results/D/scientific-result.json) | [Features](data/D/prejoin/), [gold/joins](gold/D/), [configuration](config/D/) |
| E: two-stage policy checks | [E](studies/03-two-stage-policy-checks/) | [E results](results/E/scientific-result.json) | [Features](data/E/prejoin/), [gold/joins](gold/E/), [configuration](config/E/) |
| F: routing, relation, disposition | [F](studies/04-routing-relation-disposition/) | [F results](results/F/scientific-result.json) | [Frozen arrays](data/F/prediction-freeze/), [configuration](config/F/) |

The kernels, Path A generators, Path B checkers, blueprints/contracts and B1/B3
scientific files are preserved byte-for-byte. Offline adapters and frozen join
helpers are separate from unchanged kernels. [MANIFEST.json](MANIFEST.json)
contains the exact relative file list, hashes, sizes and upstream identities.
[B4-MANIFEST.json](B4-MANIFEST.json) binds the offline input overlay, including
byte-identical source members shared with the original source export.
[Frozen protocols](reproduction/protocols/) preserve the scientific study plans;
[B4-PROTOCOL-MANIFEST.json](B4-PROTOCOL-MANIFEST.json) binds their exact bytes.

[CLAIM-FIELD-MAP.json](CLAIM-FIELD-MAP.json) has 59 complete saved-field traces.
Its paper-location labels are upstream trace references, not a new audit of
a later manuscript. Figure 1 is conceptual. Source schema_version values remain
intact; reduced custody envelopes are not complete original custody schemas.

## Results and Coverage

C retains 2160 primary-arm surface rows per encoder over 720 canonical pairs.
Its encoder-specific primary gains failed the gate; NO_SEPARABLE_SIGNAL,
reader nonconvergence and diagnostic-only raw-norm limits remain visible.
Primary metrics can use the retained probabilities and gold labels. Separate
per-row lexical/shortcut control predictions are not supplied for every control.

D supplies all 2808 rendered feature rows per encoder, original thresholds,
gold labels and joins. Its 936 canonical pairs expand to 2808 surface pairs
per encoder: 936 calibration and 1872 sealed. Do not divide by encoder count.
Workable/narrow/misleading component findings do not erase the full-profile null.

E supplies all 10368 pair feature rows per encoder, original thresholds and
gold/joins: 3456 calibration and 6912 sealed pairs over 1296 actions. All eight
arms and 864 sealed case outcomes per arm/encoder remain visible. Reduced
policy evaluations did not rescue its serial-path null or accuracy/safety failures.

F supplies 14 relation probability arrays and 30 router masks, configuration,
saved confusion tables, family macro-F1/effects and comparator action outputs.
Useful routing does not erase the null incremental relation or failed
disposition finding. Universal escalation and action accuracy 840/2304 remain;
zero false ACT with no predicted ACT is not a demonstrated safety advantage.
Aggregate confusion and family summaries support selected recalculation and
summary-level inference using the frozen procedures.

An independent frozen-procedure summary check reproduced 14 exact twelve-family
sign-flip p-values, 12 Holm adjustments and two seeded bootstrap intervals at
stored twelve-decimal precision. Means reconstructed from rounded vectors
differed by at most 5.83333e-13 and family effects by at most 1.00005e-12,
within the documented quantization bounds; test p-values, adjusted p-values and
interval endpoint strings matched without tolerance. This validates the saved
summary-level inference, not its primitive-row derivation or a model rerun.

F's exact primitive ordinal/truth/family joins are not supplied, so the arrays
alone do not independently reconstruct its sampled relation confusion/family
tables. Full routing budget-match and the accepted serial comparator also need
the original F cosine/Euclidean/legacy-direction features. Complete F
representation bodies are not bundled. The exact conditional sample procedure
and required hashes are in [reproduction/README.md](reproduction/README.md) and
[sampler provenance](config/F/SAMPLER-PROVENANCE.json). No guessed joins or
truth inferred from predictions substitute for these missing inputs.

The consequence-graph hypothesis is untested by these four studies. Latency is
recorded original-environment evidence, not portable timing or cross-platform
byte equality. [REPRODUCIBILITY.json](REPRODUCIBILITY.json) records the factual
coverage and remaining technical limits for this version.

## Reader Commands

From this companion root, Python stdlib only, without scientific imports:

```sh
python3 -B tools/mechanical.py .
python3 -B reproduction/offline_reader.py verify
```

With numerical dependencies available, these supplied entry points validate
input/result hashes before calling the unchanged frozen reducers:

```sh
python3 -B reproduction/offline_reader.py C-primary --result results/C/scientific-result.json
python3 -B reproduction/offline_reader.py D-sealed --result results/D/scientific-result.json
python3 -B reproduction/offline_reader.py E-sealed --result results/E/scientific-result.json
python3 -B reproduction/offline_reader.py F-summary --result results/F/scientific-result.json
```

These commands are implemented, not evidence of successful execution of all
analyses or verification of the released adapter itself. The separate independent
F summary check used a frozen-procedure transcription. This assembly ran only
packing/manifest/field checks, not these
reducers, fitting, inference or full experimental reruns. D/E use fixed
thresholds rather than retuning calibration. Some frozen reducers execute
prespecified bootstrap/permutation procedures when readers invoke them.
Outputs describe coverage and field mismatches; a mismatch is not a successful
recalculation. Twelve-decimal saved features/probabilities can affect ties.
The adapter's numeric comparison tolerance is 1e-9, not permission to overlook
categorical differences. Construction/conditional F commands and exact exclusions
are in [the offline procedures](reproduction/README.md).

A deterministic archive can be created without scientific execution:

```sh
python3 -B tools/archive.py . ../semantic-geometry-judgement-v0.1.0-sealed.tar.gz
```

Manifest/field trace, recalculation from sufficient statistics, reconstruction
of primitive metric inputs and a full model/corpus rerun are different levels.
The first does not certify the others. Independently pin the manifest hash for
the precise version being inspected.

## Dependencies and License

Recorded D/E/F Python is 3.12. [Analysis requirements](reproduction/analysis-requirements.txt)
give the portable NumPy/SciPy resolution, not proof of the original active D/E/F
transitive environment. [Original source declarations](reproduction/environment/source-declared-requirements.txt)
and [observed C distributions](config/C/observed-distributions.json) remain
distinct from a verified environment lock; C records two filelock versions.
Model identity/revision, prefixes, pooling, precision, length limits and seed are
under config/*/model-config.json. Fresh model reproduction requires separately
admitted model assets and a resolved active environment. No dependency package,
model or weight is bundled or relicensed; upstream terms govern acquisition/use.
Exact publisher/revision attribution and current dependency terms pointers are
in [THIRD-PARTY-REFERENCES.md](THIRD-PARTY-REFERENCES.md), distinguished from
original installed-artifact terms and runtime identity. E02's configured snapshot
license is not asserted where its pinned README was unavailable.

[LICENSE](LICENSE) is the standard MIT copyright license for original InflectAI
contributions to the extent its licensable rights exist. [NOTICE](NOTICE)
reserves SRF System K patent rights and excludes third-party rights.
[CITATION.cff](CITATION.cff) supplies version/title/author metadata. No DOI,
arXiv identifier or invented live release URL is supplied. Relative links resolve
to the actual files; immutable publication links must identify the exact
authorized GitHub tag/release or commit after release verification.
