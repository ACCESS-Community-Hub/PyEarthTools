"""
Register persistence model in zoo

NOTE:

- this is temproary compatibility with pipeline ingest to fit in with the paradigm similar to
  FourCastNeXT.

- zoo may get deprecated in favour of direct implementations in bundled models, so any interfacing
  is intentionally lightweight, with some shortcuts.
"""


@pyearthtools.zoo.register("Development/Persistence", exists="ignore")
class PersistenceRM(pyearthtools.zoo.BaseForecastModel):
    _name = "Development/Persistence"

    def __init__(
        self,
        *,
        pipeline_name: str = None,
        output: Optional[os.PathLike] = None,
        pipeline=None,
        lead_time: int | str,
        **kwargs,
    ) -> None:
        """
        TODO initialize persistence class with appropriate arguments
        """
        raise NotImplementedError("TODO")
        super().__init__(
            pipeline_name=pipeline_name, pipeline=pipeline, output=output, **kwargs
        )

    def load(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        """
        TODO

            - check pipeline was constructed with a TemporalWindow or equivilent Temporal* index
              extraction methods.
            - pass the merged indices into the persistence algorithm
            - the return type should be a "Predictor" that accepts some kwargs
            - for a simplistic persistence model we don't want the recurrent predictor, as the
              internal methods already handle any splitting and stacking.
            - instead use the TimeWindow directly
            - I'm not sure how this handles data sets

        The easiest way to do this is to:

            - look at a sample pipleline with a TemporalWindow method
            - determine how to translate the variables into an output
            - standardise the output to look like the original example

        FUTUREWORK

            while predictors in other cases e.g. fourcastnext have caching implemented. The strategy
            needs to be considered carefully. So it will be bypassed for the initial implementation
        """
        raise NotImplementedError("TODO")
