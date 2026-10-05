# Third-Party References

No third-party model weights, tokenizers, datasets, runtime/dependency packages
or vendored upstream source are included or relicensed by this companion.
The root MIT license applies only to InflectAI-held original contributions.
Upstream terms and required notices govern separately acquired components.

## Frozen Encoder References

These identifiers/revisions are preserved in config/C through config/F
model-config.json. C/D/E use E01/E02/E03; F uses E01/E02. Configurations also
preserve pooling, prefixes, maximum length, precision and seed. Configuration
metadata is not a model execution, asset admission or redistribution receipt.

| Slot | Publisher/model | Exact configured revision | Upstream terms reference |
|---|---|---|---|
| E01 | sentence-transformers/all-mpnet-base-v2 | e8c3b32edf5434bc2275fc9bab85f82640a19130 | [Pinned publisher README](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/blob/e8c3b32edf5434bc2275fc9bab85f82640a19130/README.md) declares apache-2.0 in its license metadata. |
| E02 | intfloat/e5-base-v2 | f52bf8ec8c7124536f0efb74aca902b2995e5bcd | [Publisher model/terms page](https://huggingface.co/intfloat/e5-base-v2). This is a current publisher pointer, not verification of the license at the configured snapshot; the pinned README read was unavailable. No pinned-snapshot license assertion is made. |
| E03 | BAAI/bge-base-en-v1.5 | a5beb1e3e68b9ab74eb54cfd186867f64f240e1a | [Pinned publisher README](https://huggingface.co/BAAI/bge-base-en-v1.5/blob/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/README.md) declares mit in its license metadata. |

These metadata references do not grant rights on any publisher's behalf or
establish that an asset was present in an original executing image. No assets
were downloaded to assemble these references. Any later model acquisition must
check the actual revision's terms and asset identity independently.

## Declared Dependency References

The unchanged source-declared requirements name the versions below. Numerical
analysis requirements supply the NumPy/SciPy resolution separately. Version
declarations and observed distributions are not proof of original active
installed artifacts or a complete resolved runtime. These official links are
current upstream terms pointers, not immutable historical installed-artifact
license attestations; actual obtained distributions may carry additional notices.

| Dependency | Original source declaration | Official current license pointer |
|---|---|---|
| NumPy | 2.5.3 | [NumPy license](https://numpy.org/doc/stable/license.html) |
| SciPy | 1.18.1 | [SciPy LICENSE.txt](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| PyTorch / torch | 2.14.0 | [PyTorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| Transformers | 5.16.1 | [Transformers LICENSE](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| huggingface-hub | 1.30.0 | [huggingface_hub LICENSE](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) |
| safetensors | 0.8.0 | [safetensors LICENSE](https://github.com/safetensors/safetensors/blob/main/LICENSE) |

The recorded C inventory includes two filelock versions and is not an
installable lock. Original D/E/F active transitive versions remain unestablished.
No upstream dependency is redistributed under the companion's MIT notice.
