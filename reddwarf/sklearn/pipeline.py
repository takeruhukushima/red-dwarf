from sklearn.pipeline import Pipeline


class PatchedPipeline(Pipeline):
    """
    A subclass of sklearn's Pipeline that injects a `_parent_pipeline` attribute into each step.

    This allows individual transformers in the pipeline to access their parent pipeline and,
    by extension, other steps within it. Useful for custom transformers that depend on
    intermediate results from earlier steps (e.g., SparsityAwareScaler using SparsityAwareCapturer output).

    Example:
    ```
    pipeline = PatchedPipeline([
        ("capture", SparsityAwareCapturer()),
        ("scale", SparsityAwareScaler(capture_step="capture")),
    ])

    # Inside SparsityAwareScaler.transform():
    # capture_step = self._parent_pipeline.named_steps["capture"]
    # X_sparse = capture_step.X_transformed_
    ```

    Note:
        - Steps must support attribute assignment (`__dict__`) to receive the reference.
        - `_parent_pipeline` is injected once during initialization.
    """
    def __init__(self, steps, **kwargs):
        super().__init__(steps, **kwargs)
        self._patch_steps()

    def _patch_steps(self):
        for _, step in self.steps:
            if hasattr(step, '__dict__'):
                step._parent_pipeline = self