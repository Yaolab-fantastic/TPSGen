# Model migration audit

This internal document records the evidence used to migrate the three thesis
mechanisms into TomatoPromoterDesigner. It is not manuscript text. A component
is considered released only after its source, checkpoint, input data and output
have been matched and tested together.

## Release rule

Each mechanism must pass all of the following checks before it is described as a
package capability:

1. The implemented architecture strictly loads the intended checkpoint.
2. Input preprocessing matches the preprocessing used for that checkpoint.
3. A retained result can be reconstructed from a documented command.
4. Output fields state whether values are model scores, measured values or
   descriptive sequence statistics.
5. The manuscript, Supplement, README and CLI use the same mechanism name.

## Mechanism 1: TransVAE-MLP

### Thesis definition

The Chapter 3 mechanism combines a Transformer-based VAE with a four-output MLP.
The same latent representation supports sequence reconstruction/generation and
root, stem, leaf and fruit target estimation. The thesis then describes latent-
space search for fruit-biased candidates.

### Matched original resources

| Resource | Original path | Verified observation |
|---|---|---|
| Transformer building blocks | `MpraVAE/code/TransVAE.py` | Contains Transformer encoder/decoder, convolutional bottleneck and autoregressive decoder components. |
| Joint training logic | `MpraVAE/code/transformervae.py` | Contains four-target loading, reconstruction/KL/prediction terms, KL warm-up and retained training metrics. |
| Transformer checkpoint | `MpraVAE/code/transformerresult/models1/best_val_corr_model.pth` | Contains 180 keys, including `transvae.*`, `predictor.*` and `vocab_to_base.*`. |
| Training history | `MpraVAE/code/transformerresult/results1/loss_curves/loss_history.csv` | Contains 50 epochs of retained losses. |
| Training log | `MpraVAE/code/transformerresult/logs1/training_log.log` | The best mean validation correlation occurs at epoch 47: 0.7801, 0.8237, 0.8019 and 0.7924 for targets 1--4. |
| Retained Transformer candidates | `MpraVAE/results/trans_designed_promoters200.csv` | Contains 20 retained candidate rows. |

### Blocking inconsistencies

- The checkpoint uses a 128-dimensional latent representation: for example,
  `transvae.encoder.z_means.weight` has shape `(128, 896)` and the predictor
  begins with a `(128, 128)` weight. It is not the 64-dimensional convolutional
  checkpoint currently bundled by the package.
- The checkpoint contains a 23-token embedding and a 22-token generator output.
  The retained vocabulary file is a peptide vocabulary inherited from the base
  Transformer implementation, while the joint training script converts DNA
  one-hot channels directly to integer indices using `A=0`, `C=1`, `G=2` and
  `T=3`. The unused embedding and generator dimensions are therefore inherited
  implementation capacity, not 23 biologically defined DNA tokens.
- The joint wrapper fragment sets `pad_idx=0` and constructs masks from
  `token != 0`, although token 0 also represents A. Consequently, the checked
  source masks adenine positions as if they were padding. This behavior is a
  retained implementation defect, not a biologically motivated preprocessing
  choice. Compatibility inference and any corrected retraining route must keep
  these two behaviors explicit and separate.
- The checked `transformervae.py` calls `JointPromoterModel`, but that class is
  absent from the file. A second `forward` method is incorrectly nested under
  `ExpressionPredictor`. The source therefore cannot currently recreate the
  checkpoint architecture without reconstruction.
- The script writes `params['latent_dim']`, whereas `create_VAE` reads
  `params['d_latent']`. The retained checkpoint demonstrates that the effective
  latent dimension was 128.
- The reported 3-mer term is calculated from detached, discretized outputs and
  wrapped in a new tensor. It is logged in the total objective but does not
  propagate a gradient to the generator. It must not be described as an active
  differentiable training constraint without correction and retraining.
- `generate1.py` performs genetic-algorithm search using the separate
  convolutional `vaecnn1.JointPromoterModel`; it is not a verified design entry
  point for the 180-key Transformer checkpoint.
- `tragenerate.py` is not an executable Transformer design route. It calls a
  nonexistent `transvae.embedding`, assumes incompatible encoder and decoder
  signatures, passes undefined keyword arguments and later calls `model.decoder`,
  which is not present on the recovered joint wrapper.

### Checkpoint-to-data verification

A clean reconstructed architecture strictly loads all 180 checkpoint keys. The
missing joint wrapper was recovered from the retained Python 3.10 bytecode. This
showed that the fourth value returned by `TransVAE.forward` was misleadingly
named `z` in the wrapper: it was actually
`predict_len2(predict_len1(mu))`, and this 128-dimensional deterministic
representation was passed to the expression MLP. The predictor did not receive
the sampled latent variable.

Using the recovered wrapper behavior, source-defined DNA mapping, the first 164
positions of each 165-bp sequence and the retained `token != 0` mask, the
checkpoint reproduces the best validation-log values on `val.csv` (3,703 rows).
Computed correlations are 0.780106, 0.823665, 0.801869 and 0.792406; these match
the rounded epoch-47 log values 0.7801, 0.8237, 0.8019 and 0.7924. The separate
`validation_set.csv` contains 3,000 rows and is not the validation table used for
this retained checkpoint.

This establishes a checkpoint/data/code pairing for reconstruction. It does not
remove the methodological limitations of using token 0 for both adenine and
padding, nor does it make the retained validation split an independent external
benchmark.

### Design-path verification

The thesis describes genetic-algorithm search within a plus/minus 2 neighborhood
of an encoded latent vector. Its fitness is the fruit score minus the maximum of
the root, stem and leaf scores, with a penalty for negative non-fruit scores.
That scientific target is clear, but the retained Transformer checkpoint does
not currently provide a valid implementation of the stated design path.

The historical joint wrapper shifts the DNA tokens into `src=token[:-1]` and
`tgt=token[1:]`, but its target mask only removes token 0 positions. It does not
apply a causal autoregressive mask. Under this retained teacher-forced path, the
decoder reconstructs 99.83% of bases in a 128-sequence diagnostic batch. However,
replacing the decoder latent input with zeros, random vectors, reversed-sample
means or the MLP prediction representation leaves 100% of decoded base calls
unchanged. The decoder is therefore copying information from the unmasked target
context and is not demonstrably conditioned on the latent representation.

The retained `trans_designed_promoters200.csv` is also not releasable evidence:
its 20 generated sequences contain on average 92.36% C and no G or T, with a
median maximum homopolymer length of 103 bp (range 48--143 bp). These sequences
are degenerate and fail basic sequence-quality expectations.

Consequently, the package must not expose latent-space GA design from this
checkpoint. A valid route requires corrected causal decoding, a single explicitly
shared representation for the predictor and decoder, retraining, and validation
that decoded sequences both respond to latent perturbation and pass predefined
sequence QC. The thesis fitness function can be retained in that corrected
training/design implementation, but the historical candidate table cannot be
used as its validation output.

### Current package state

The current `models/mpravae/best_val_corr_model.pth` is a 51-key convolutional
VAE checkpoint. The matching model is implemented in
`legacy/transvae_tomato.py`. This route must be treated as a convolutional VAE
backend and must not be presented as the complete thesis TransVAE-MLP mechanism.

### Migration decision

Status: **blocked from release, available for reconstruction work**.

A clean 128-dimensional Transformer joint core has now been reconstructed,
strictly loads all 180 keys and reproduces the retained epoch-47 validation
correlations. It remains internal until the checkpoint is packaged, the
compatibility limitations are represented in user-facing metadata and the
Transformer-compatible design route is retrained and validated. A corrected
architecture needs an explicit DNA vocabulary and padding token, causal target
masking and a shared predictor/decoder representation; it must be released as a
distinct model version. The existing convolutional route will remain available
under an explicit backend name during this migration.

## Mechanism 2: DNABERT-derived motif identification

### Matched resources

- Base 6-mer checkpoint: `DNABERT/6-new-12w-0.zip`.
- Matching retained attention input: `DNABERT/examples/sample_data/vision/dev.tsv`.
- Retained arrays: `DNABERT/examples/result/6/atten.npy` and
  `pred_results.npy`, both with 10,222 rows.
- Retained fine-tuned-model candidate: `DNABERT/examples/ft/6`, containing a
  12-layer, 768-hidden-unit, 12-head `BertForSequenceClassification` checkpoint
  with a two-class head and `finetuning_task=dna690`.
- Motif processing: `DNABERT/motif/find_motifs.py` and `motif_utils.py`.
- Mutation processing: `DNABERT/SNP/`.

The 10,222-row `vision/dev.tsv` is byte-identical to
`tomatoft/tomatoft3-qvjian/train.tsv`. The separate 6,000-row
`tomatoft/dev.tsv` does not match the retained attention array.

The checkpoint in `examples/ft/6` is not the model that produced the retained
arrays. Recomputing class-1 probabilities for the first ten rows of
`vision/dev.tsv` gives a maximum absolute difference of 0.932 from
`examples/result/6/pred_results.npy`. A retained cache filename,
`cached_dev_fitresultnew_81_dnaprom`, indicates that the visualization run used
a checkpoint or output directory named `fitresultnew`; that checkpoint is not
present in the checked tree. The existing `ft/6` model must therefore not be
silently substituted.

Status: **post-processing is reconstructable; raw FASTA-to-attention inference
requires the matching `fitresultnew` tomato fine-tuned checkpoint, which has not
yet been located**. The unrelated `ft/6` classifier is retained evidence of a
separate fine-tuning run, not a valid replacement.

### Post-processing reproduction

The package adapter now follows the retained operation order: exact
high-attention subsequences are tested for enrichment first, Benjamini-Hochberg
adjustment is applied across those exact motifs, and only significant motifs are
then merged and expanded to 24-bp windows. The previous adapter incorrectly
merged motifs before testing and returned the 20 highest-ranked candidates when
none was significant. That fallback converted adjusted p-values of 1.0 into
apparent motif output and has been removed. An empty significant set now produces
an explicit empty-result record rather than nonsignificant motifs.

Using `vision/dev.tsv` (10,222 rows), `result/6/atten.npy`, `window_size=24`,
`min_len=5`, adjusted-p-value cutoff 0.005 and `min_n_motif=3`, the corrected
adapter extracts 8,096 exact motifs, retains 787 after enrichment testing,
produces 77 merged motif groups and retains 53 motif records after 24-bp
windowing and minimum-instance filtering. The regenerated repository table
reports the top retained motifs as ACTATA (127), CTCAAA (126), TAATTT (96),
ACTTAT (95) and TTAAA (90). Older retained `6-2` and `6-3` summary files are
byte-identical frequency-style retained copies and must not be described as
independent validation runs or as the current adapter output.

Exact reconstruction requires Biopython 1.85 or later. Biopython 1.79 selects
different optimal terminal-gap alignments in `PairwiseAligner`, yielding 29
motifs and altered instance counts on the same data. The package therefore sets
`biopython>=1.85`. The retained script's `align_all_ties` branch compares
alignment objects rather than their numeric scores; the compatibility path
preserves that actual behavior because changing it duplicates instances across
equally scoring canonical motifs. This is a provenance-compatible reconstruction
of retained attention outputs, not FASTA-to-attention inference and not an
independent biological validation of the DNABERT classifier.

## Mechanism 3: preGAN

### Matched resources

- Conditional data: `deepseed/data/merged_result.csv`, containing 5,100 rows of
  `realA`, `realB` and `expr`.
- Mask encoding: `deepseed/Generatorme/pro_data.py`.
- DenseLSTM definition and checkpoint: `deepseed/Predictor/SeqRegressionModel.py`
  and `deepseed/Predictor/results/model/165_mpra_expr_denselstm.pth`.
- Closest predictor-guided WGAN-GP implementation:
  `deepseed/Generator/preGAN.py`.

`pre-GAN-target.py` does not call a predictor in the expression term.
`pre-GAN-target1.py` instantiates a DenseLSTM predictor but does not load trained
weights. These files are experimental variants and will not be migrated as the
released implementation.

Status: **partially reconstructable**. The predictor checkpoint exists, but the
referenced preGAN generator checkpoint has not yet been located. The corrected
training route must be parameterized and tested; otherwise the generator must be
retrained before preGAN inference is released.

### Predictor and training-path audit

Only two model checkpoints were found under the complete `deepseed` tree, both
named `165_mpra_expr_denselstm.pth`. They deserialize as complete DenseLSTM
models with 612,897 parameters and 175 state tensors, but they are not copies:
none of their 175 tensors is identical. The packaged checkpoint has SHA-256
`fec9ae772a4ca39635b25e7fdb7dddbd2d4d83b333979276701a680dac60c86c`
and exactly matches `Predictor/results/model/165_mpra_expr_denselstm.pth`. Its
retained training log ends at a validation correlation of approximately 0.710.
The alternative `results1` checkpoint has SHA-256
`d1ddc57f11b280ac302ccaaf832a0eaa3f3066be93e823422edcfca538993b51`
and its retained log ends near 0.021 with much larger losses. The `results1`
branch must not supply figures for the packaged `results` checkpoint.

The corresponding candidate training table `data/output.csv` contains 8,545
165-bp sequence/expression rows. The conditional GAN table
`data/merged_result.csv` contains 5,100 rows with `realA`, `realB` and `expr`;
`realA` uses `M` to mark mutable positions while fixed nucleotides preserve the
retained motif/flanking context. These row counts describe retained resources,
not independent test cohorts.

None of the retained GAN scripts forms a releasable predictor-guided training
route as written. `preGAN_expr.py` accepts a predictor argument but does not use
it in its loss, and its entry point passes no predictor. `pre-GAN-target.py`
compares the generated sequence tensor with a tensor filled with 100 rather than
calling an expression model. `pre-GAN-target1.py` does call a DenseLSTM in the
loss, but instantiates it with random weights and never loads either retained
checkpoint. No generator or discriminator checkpoint is present anywhere under
`deepseed`. Consequently, the package may expose the matched DenseLSTM scalar
scorer, but must not expose trained preGAN generation until a corrected training
implementation is retrained and a matching generator checkpoint is validated.
