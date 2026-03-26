"""
Register persistence model in zoo

Here a PhantomPredictor is used as a proxy for Predictor. A persistence model is a _light_
predictor. A persistence model is a _ligh_ model.

_Light_ here implies that the model only needs the reference to the data pipeline and some
"Metadata" about the persistence model. The predictor only needs to run the "predict" function under
the assumption that the datapipeline ingested as part of the model _must_ be invertible or an
equivilent consistent inversion be provided to all predictors running aside the prediction pipeline
(if any).
"""

import datetime
import pyearthtools.training as pet_train
import pyearthtools.pipeline as pet_pipe
import typing

Prediction = typing.Any


class PhantomPredictor:
    def predict(*args, **kwargs) -> Prediction:
        """
        INTERNAL ONLY - this is aimed at developers not users

        Read the comments on `PhantomPredictor.register(...)` below first.

        This is NOT a abstract method, and it SHOULD NOT _ever_ be a abstract method. A phantom
        predictor is a imaginary concept. It is simply being used as a gateway/proxy for
        initializing a predictor without arbitrary impositions. To put it precisely:

            A Predictor is a category of THINGS that uses THINGS to _predict_ THINGS

            The THINGS it _predicts_ is called a Prediction.

        Don't let the word THINGS throw you away, it is an abstraction. Symbolically this is what is
        happening.

            * :: (* -> *)

        Contrived as it may seem every part of this docstring above is REQUIRED information for
        consideration because this is the impersonated ROOT of any Predictor. IT can't know ahead
        just like a Car can't know ahead if it will always have an engine. But it can know about
        what it conceptually SHOULD represent at all times and that is

            1. A Predictor is a named collection of things called `Predictor`
            2. which have the ability to apply a specific named operation `predict`, to a
            arbitrarily structured input
            3. and produce a namespaced output called `Prediction`.

        With no bounds in what they can actually mean.

        >>> p = PersistencePredictor(PhantomPredictor)
        >>> assert isinstance(p, Predictor)
        ... # Note that if we get past this initialization was successful
        >>> arr_in = np.array(...)
        >>> arr_out: Prediction = p.predict(arr_in)

        ```
        """
        raise NotImplementedError("A predictor MUST know how to predict things.")


class PhantomModel:
    def get_model(*args, **kwargs) -> PhantomModel:
        """
        Similar to predictor this is not a abstract method. But it is mandatory. A model can exist,
        but until it returns a tangible version of itself it is a ghost.
        """
        raise NotImplementedError(
            "A model is a RESOURCE, a resource MUST be shareable, but it DOES NOT NEED to be shared."
        )


# ---
# Phantom classes: Do not contain data. They exist as an abstraction layer to remove class
# "impositions" such as abstractmethods. This is a flaw of object oriented design as opposed to type
# oriented design.
#
# Object-Oriented: A Car has a engine (pre EV)   <--- defines somethings by its functionality
#                  Implement Car for ElectricCar <--- entire framework falls apart, because functionality has changed
#
# Type-Oriented:   A Car is a Car <-- defines something as literally what it is, universal, never fails
#
# ```
# >>> class MyPredictor(PhantomPredictor): pass
# >>> x = MyPredictor()  # <--- works
# >>> isinstance(x, Predictor)
# ... True
# >>> class MyPredictorOrig(Predictor): pass
# >>> y = MyPredictorOrig()  # <--- does not work
# ... Exception...
# ```
# The above IS a hack, but a necessary evil barring a big refactor.
# ---
pet_train.Predictor.register(PhantomPredictor)
pet_train.Model.register(PhantomModel)


@pyearthtools.zoo.register("Development/Persistence", exists="ignore")
class PersistenceRM(PhantomPredictor):
    """
    The persistence model is a temporal predictor.

    The concept of "time" is a bound criteria. In particular, Persistence is a causal predictor
    and therefore can only predict use historical data strictly before a reference time and only
    produce future reuslts strictly greater than or equal to a reference time. With the affordance
    that the reference time is usually the first future timestep.

    The input requirements or THINGS required to perform a prediction
    1. the pipeline - which is user defined, common to both the persistence model and dictates how
       the loading and preprocessing happens, which usually is, but not always expected to be the
       same as the pipeline of whatever model is running in parallel.
    2. the reference time.
    3. the model: is actually a true phantom data. A persistence method is not "modelled" in anyway
       it has no "brains" or thought process behind it past heuristic and a null hypothesis
       formation. Therefore the model IS the predictor, and the resource associated to the model IS
       the cached set of metadata required to do the prediction

    Since this is usually the entry point, the class initalizer should ingest standard arguments
    instead of internal structures.

    Args:

        pipeline:       The data pipeline

        dt_base:        The start time of the forecast, though its coined "start time" really in a
                        lot of persistence models it its the _only_ time step returned. However,
                        what the start_time does is give the underlying method a reference to
                        understand where it can extract additional historical data from for
                        on-the-fly statistical leanring.

        method:         The persistence method to be computed; defaults to MEDIAN_OF_THREE (as enum
                        or "median_of_three" as string).  For speedier compute, you should use
                        MOST_RECENT. There are fancier algorithms planned that may be slower but
                        worth it for the accuracy gain (within reason).

        name_time:      The name of the time dimension.

        num_threads:    The number of threads to allocate (usually set this to 1, but in some
                        systems it may help increasing this to make I/O faster.

        num_chunks:     Chunking is useful to have even without multiprocessing or multithreading
                        because it can provide guarentees on how much data is being accessed if its
                        done consistently.

        simple_impute:  Whether to impute the data. Usually simple and fast enough that its trivial
                        to do, but sometimes the backend algorithm may already be doing it in which
                        case its worth it to set it to false.

        backend_type:   which backend to use to do the actual work (usually "numpy" but "zig" is
                        also supported as an experimental option).


    """

    # TODO: finish typehints for the folowing
    data_pipeline: pet_pipe.Pipeline
    dt_base: datetime.datetime
    pipeline
    dt_base
    method
    name_time
    num_threads
    num_chunk
    simple_impute
    backend_type

    _name = "Development/Persistence"

    def __post_init__():
        """
        Dynamically define the model which is essentially a wrapper around persistence_impl.py with
        the fit operator pre-populated with the class entries.
        """
        self.model = ...  # TODO: model needs to be defined with "Fit Function"

    def predict():
        """
        A predictor can only be used if it has a predict() method - this is part of the contract
        """
        # need some pre logic to recreate the array

        self.model.fit(...)

        # need some post logic checks


# TODO: use
class PersistenceModel(
    PhantomPredictor
): ...  # TODO: Define a modle as a callable id model = PersistenceImpl(...) <--- this is the model resource


if _name__ == "__main__":
    predictor = PhantomPredictor()
    model = PhantomModel()
    # Quick test of impersonation - this should never fail
    assert isinstance(predictor, pet_train.PhantomPredictor)
    assert isinstance(model, pet_train.PhantomModel)
