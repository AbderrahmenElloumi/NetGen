# NetGen's own learning-stage protocol instead of a scikit-learn Pipeline

NetGen's learning stages (the fill step, the scaler and the optional autoencoder) follow a small protocol of NetGen's own: `fit(table)` returns a frozen fitted object, which transforms tables and saves its state as plain JSON (or weights-only torch state for the autoencoder) in the artefacts folder. A single `preparation` module fixes their order. The obvious alternative was scikit-learn's Pipeline and ColumnTransformer, which reviewers know well. We rejected it because its fitted state saves through pickle, which the artefacts contract forbids, so the numbers would have to be extracted to JSON by hand anyway; because column roles and derived columns (the anomaly score and latent columns) fit awkwardly into ColumnTransformer's column selection; and because scikit-learn may not otherwise be a dependency once the validation report is decided. Library math may still be used inside a stage.

## Considered options

- scikit-learn Pipeline and ColumnTransformer: rejected for pickle-based state, awkward role handling, and the extra dependency.
- A scikit-learn-style object whose `fit` changes it in place: rejected because half-fitted objects and "transform before fit" bugs become possible.
- A user-configured pipeline of stages: rejected because it allows orders that leak test rows into fitting, and nothing needs that variation.
