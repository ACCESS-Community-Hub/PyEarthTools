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

import dataclasses
import functools
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

        With no bounds in what they can actually encompass.

        >>> p = PersistencePredictor(PhantomPredictor)
        >>> assert isinstance(p, Predictor)
        ... # Note that if we get past this initialization was successful
        >>> arr_in = np.array(...)
        >>> arr_out: Prediction = p.predict(arr_in)

        ```
        """

        # ---
        # NOTES:
        # Don't let the word THINGS throw you away, it is an abstraction. Symbolically this is what is
        # happening.
        #
        #     * :: (* -> *)
        #
        # Contrived as it may seem every part of this docstring above is REQUIRED information for
        # consideration because this is the impersonated ROOT of any Predictor. IT can't know ahead
        # just like archetype Car can't know ahead if it will always be required to have an engine.
        # But it can know about what it conceptually SHOULD represent at all times and that is
        #
        #     1. A Predictor is a named collection of things called `Predictor`,
        #     2. containing maps that `predict`
        #     3. and produce a namespaced output called `Prediction`.
        # ---
        raise NotImplementedError("A predictor MUST know how to predict things.")


class PhantomModel:
    def get_model(*args, **kwargs) -> PhantomModel:
        """
        INTERNAL ONLY - this is aimed at developers not users

        Similar to predictor this is not a abstract method. But it is mandatory. A model can exist,
        but until it returns a tangible version of itself it is a ghost. Without a concrete
        definition of get_model that is ingestable by some other entity - its existence (or promised
        existence) is meaningless.
        """
        # ---
        # NOTES:
        # does not need to be shared => there is a guardrail that is YET to be determined
        # usually by e.g. a predictor
        #
        # it must be shareable => some entity needs to refer to the tangential model (data, e.g.
        # wiehgts) not the conceptual model (the contract or the class Model), for it to properly
        # exist.
        #
        # these are again just _contracts_
        # ---
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


# ---
# IMPORTANT: models DO NOT need to be defined like this. But where they fit the mould exactly, they
# SHUOLD be defined like this, i.e. a static model with prediction capability ingrained.
#
# Usually this dependency would be reversed, because a model is pre-defined, in persistence the
# model does not exist until the user provides data and the method to compute it.
#
# Therefore, as far as persistence models are concerned you can just directly use the
# PersistencePredictor if you are using this as a library.
#
# TLDR;
# 1. this is just yet another namespacing issue from legacy code;
# 2. Persistence is a special case where models are predictors.
#
# RM => RegisteredModel as per usual
# ---
@dataclasses.dataclass(frozen=True)
@pyearthtools.zoo.register("Development/Persistence", exists="ignore")
class PersistenceRM(PhantomModel, PhantomPredictor):
    """
    This is the main entry point to the registered model that computes persistence. There is a
    caveat here. Even though this is the main entry point, this is mainly for consistency. It is
    completely interchangeable with using the PersistencePredictor directly which has a richer
    documentation.

    See: `PersistencePredictor` for a more detailed information on the arguments.

    This is just a compatiblity layer to adhere to registered models.
    """

    _name = "Development/Persistence"

    # TODO: if there are docstrings issue, the user should just be referred to PersistencePredictor
    def get_model(self) -> "PersistenceModel":
        """
        NOTE: usually this returns "data" that a external predictor can use, but a persistence
        model's representation of data is itself. So contrived as it looks, this is an accurate
        definition. But as specified in PhantomModel this part of the contract while, MUST be
        defined, _DOES NOT NEED_ to be used.

        This part of the code was written _after_ the PhantomModel, so if you are wondering why that
        part of the contract exists, this is why.
        """
        # TODO: ensure this is a "view" not a copy, it most likely is a view.
        return self


@dataclasses.dataclass
class PersistencePredictor(PhantomPredictor):
    """
    The persistence predictor is a temporal predictor. It uses a persistence model with a method
    that the user provides to perform a prediction. Since the model computes this at runtime, it too
    is a predictor. To see this consider that the persistence model will (most of the time) NOT fit
    the mould of: train -> store -> retrieve weights.

    The concept of "time" is a bound criteria. In particular, Persistence is a causal predictor
    and therefore can only predict use historical data strictly before a reference time and only
    produce future reuslts strictly greater than or equal to a reference time. With the affordance
    that the reference time is usually the first future timestep.

    Core assumption:
        - All variables that will be interrogated MUST have a concept of "time"
        - That concept of "time" must be CONSISTENT (e.g. shape, format etc.)
        - There are no guarentees or checks done for this, because it is impossible to do so without
          considering every edge case. this is a user requirement in sanitizing the data to conform.
        - and its slow to do so...

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

        method:         The persistence method to be computed; defaults to MEDIAN_OF_THREE (as enum
                        or "median_of_three" as string).  For speedier compute, you should use
                        MOST_RECENT. There are fancier algorithms planned that may be slower but
                        worth it for the accuracy gain (within reason).

        dimname_time:   The name of the time dimension.

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

    pipeline: pet_pipe.Pipeline
    method: str
    dimname_time: str

    _: KW_ONLY

    num_threads: int = 1
    num_chunks: int = 1
    simple_impute: bool = True
    backend_type: str = "numpy"

    # NOTE: not caching this yet, since there's no guarentee that this class is frozen.
    def indexofdim_time(self) -> int:
        return list(ds_input.dims).index(dimname_time)

    def predict(
        self, var_names: list[str], dt_base: datetime.datetime | str
    ) -> Prediction:
        """
        A predictor can only be used if it has a predict() method - this is part of the contract

            var_names: the variable names/keys to predict
            dt_base: the base time to use for prediction
        """
        # ----------------------------------
        # --- FULL external-backend flow ---
        # ----------------------------------
        # TODO: implement branching here to use the listener method with zmq+zig to do out of
        # process computation of the methods using custom loaders
        # ---
        # ...ZMQ magic here if requested
        # ---

        # ------------------------------------
        # --- python-native flow (default) ---
        # ------------------------------------
        # (with optional backend flow)
        # get data instance from pipeline (assume that entries are pre-concatted)
        ds = pipeline[dt_base]

        # --- perform checks ---
        # ds should ALWAYS be a dataset if we are using the pipeline
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                "PersistencePredictor: Pipeline did NOT retrieve a xr.Dataset. This is a critical failure."
            )

        if not isinstance(var_names, list) or any(
            map(lambda v: not isinstance(v, str), var_names)
        ):
            raise TypeError("PersistencePredictor: var_names MUST be a list of strings")

        # --- extract variables ---
        # select variables of interest
        # TODO: not sure if this the best way to select a view
        ds_input = ds[[var_names]]

        # --- cache coordinates to reintroduce post compute ---
        coords_out = copy.deepcopy(ds_input.coords)
        # persistence model only returns a single time index
        del coords_out[dimname_time]

        # --- run prediction ---
        # TODO num_workers -> num_threads
        ds_prediction = persistence_impl.predict(
            ds_input,
            idx_time_dim=self.indexofdim_time(),
            num_workers=self.num_threads,
            num_chunks=self.num_chunks,
            method="median_of_three",
            simple_impute=self.simple_impute,
            backend_type=self.backend_type,
        )

        # --- re-assign stripped coordinates ---
        ds_prediction = ds_prediction.assign_coords(coords_out)

        return ds_prediction


if _name__ == "__main__":
    predictor = PhantomPredictor()
    model = PhantomModel()
    # Quick test of impersonation - this should never fail
    assert isinstance(predictor, pet_train.PhantomPredictor)
    assert isinstance(model, pet_train.PhantomModel)
