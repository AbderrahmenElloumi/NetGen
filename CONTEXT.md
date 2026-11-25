# NetGen

NetGen turns network-traffic recordings into tabular datasets for machine learning, together with saved artefacts that apply the same treatment to new recordings.

## Language

### Data

**Capture**:
A recording of packets (pcap or pcapng) that NetGen turns into flows.
_Avoid_: pcap file, packet-level input

**Flow export**:
A file that already holds flows, made by another tool (today only CICFlowMeter CSV).
_Avoid_: Pre-aggregated input, NetFlow CSV

**Flow**:
One conversation between two endpoints, summarised as one row of numeric features.
_Avoid_: Connection, session, record

**Flow record**:
The fixed set of fields that every reader fills for each flow, whatever its source, with one meaning and one unit per field (times in seconds). The label is present only when the input has labels; derived columns are added later by a stage.
_Avoid_: Row schema, common set, CIC features

**Flow table**:
The flows of one input or split, with the role of each column. Never changed in place; a stage returns a new one.
_Avoid_: Frame, dataset (reserved for what a transform writes)

**Flow settings**:
The idle and active timeouts that decide how packets group into flows. Given at fit, reused by transform, ignored by a flow export.
_Avoid_: Flow options, aggregation settings

**Column role**:
What a field is for: label, identity, feature, or derived. Declared once for the whole run and recorded with the artefacts, so fit and transform always agree.
_Avoid_: Column type, column kind

**Identity column**:
A field that says which flow a row is (addresses, ports, protocol name, start time). Kept in the dataset for tracing; never scaled and never learned from, unless an option promotes the destination port to a feature.
_Avoid_: Metadata, key column

**Feature**:
A numeric field that the scaler and the autoencoder learn from.
_Avoid_: Attribute, variable

**Derived column**:
A column that a fitted stage computes from the features, such as the anomaly score or the latent columns. Written as computed; never rescaled, and no learning stage learns from it.
_Avoid_: Engineered feature, extra feature

**Anomaly score**:
How poorly the autoencoder rebuilds a flow: high means the flow looks unlike the flows it was fitted on. A derived column.
_Avoid_: Reconstruction error (the computation, not the meaning), outlier score

**Latent columns**:
The autoencoder's compressed summary of a flow's features, written only when asked for at fit time. Derived columns.
_Avoid_: Embeddings, encoded features

**Start time**:
When the first packet of a flow was seen. An identity column, never a feature.
_Avoid_: Timestamp

**Flow key**:
The two endpoints (an address, plus a port where the protocol has ports) and the protocol that decide which flow a packet belongs to, regardless of direction.
_Avoid_: 5-tuple, connection id

**Forward**:
The direction of the first packet of a flow; the other direction is backward.
_Avoid_: Upstream, outbound

**Idle timeout**:
How long a flow may stay silent before it ends; the next packet with the same flow key starts a new flow.
_Avoid_: Flow timeout

**Active timeout**:
The longest a flow may last before it is cut into a new flow, even while packets keep arriving. Off unless set.
_Avoid_: Max duration

**Label**:
The optional column that says what a flow is (for example normal or a named attack). Carried through untouched by the preparation; never a feature and never rescaled.
_Avoid_: Class, target, attack_cat

**Missing value**:
A value a reader marks as absent: a negative, infinite or non-numeric value, a rate over zero duration, or a flow-export ratio over zero forward packets. The fill step replaces it.
_Avoid_: NaN, null

**Warning**:
A note that a reader, a stage or the artefacts give about the input, shown in the console summary. It does not stop the run.
_Avoid_: Error (an error stops the run), log message

### Processing

**Reader**:
The part of NetGen that turns one input format (a capture or a flow export) into flows in the flow record. A reader learns nothing.
_Avoid_: Parser, loader

**Learning stage**:
A step that learns from the training split at fit time and repeats what it learned at transform time. There are three: the fill step, the scaler and the autoencoder.
_Avoid_: Transformer, estimator

**Fill step**:
The learning stage that replaces missing values with the median, and any infinite value that still reaches it with the largest finite value, both learned per feature from the training split.
_Avoid_: Imputer, cleaning

**Scaler**:
The learning stage that puts every feature on a comparable scale, learned from the training split: an optional log step, then standardisation to mean 0 and standard deviation 1.
_Avoid_: Normaliser

**Log step**:
The logarithm applied before standardisation to heavy-tailed features (counts, sizes, durations, gaps, rates), so that a few huge flows do not dominate. Whether a field takes it is declared once, in the flow record.
_Avoid_: Log transform, log scaling

**Constant feature**:
A feature with the same value in every training row. It is kept and written as 0, and the report names it.
_Avoid_: Dead column, zero-variance column

**Preparation**:
The fixed order of learning stages (fill step, scaler, then the optional autoencoder) that fit learns and transform repeats.
_Avoid_: Pipeline

### Commands and outputs

**Fit**:
Learning the preparation (fill step, scaler and the optional autoencoder) from training flows only, and saving it as artefacts.
_Avoid_: Train (reserved for the training split)

**Transform**:
Applying saved artefacts to flows to produce a dataset.
_Avoid_: Apply, predict, infer

**Build**:
The one-command route: split flows into training and test, fit on the training split, then transform both splits.
_Avoid_: Run, pipeline

**Artefacts**:
The folder that one fit writes: what each learning stage learned, the settings needed to repeat it, and the fit's report. Readable and safe to load.
_Avoid_: Model, checkpoint, bundle

**Manifest**:
The file in the artefacts folder that records the column roles, the feature list, the settings and the learning stages of one fit.

**Training split** / **Test split**:
The two disjoint parts of the flows that a build creates before anything is fitted; only the training split is fitted on.
_Avoid_: Train set, holdout

**Dataset**:
The table of flows produced by a transform.
_Avoid_: Output, CSV

**Report**:
The record every command writes of what it saw and did: counts of rows, labels, duplicates, filled values, dropped non-IP packets and out-of-range values, and statistics of the features and, when the autoencoder ran, the anomaly score after preparation. Observations only, never learned state.
_Avoid_: Validation, quality report, log

**Benchmark**:
An optional check, asked for at build time, that trains a classifier on the training split and scores it on the test split, to show how learnable the labels are. Its result goes in the report; it is never saved as artefacts.
_Avoid_: Validation, evaluation, accuracy
