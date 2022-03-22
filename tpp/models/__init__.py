from argparse import Namespace
from typing import Dict
from torch import nn
from pprint import pprint

from tpp.models.base.enc_dec import EncDecProcess
from tpp.models.base.modular import ModularProcess
from tpp.models.poisson import PoissonProcess

from tpp.models.encoders.base.encoder import Encoder
from tpp.models.encoders.gru import GRUEncoder
from tpp.models.encoders.identity import IdentityEncoder
from tpp.models.encoders.mlp_fixed import MLPFixedEncoder
from tpp.models.encoders.mlp_variable import MLPVariableEncoder
from tpp.models.encoders.stub import StubEncoder
from tpp.models.encoders.self_attention import SelfAttentionEncoder

from tpp.models.decoders.base.decoder import Decoder
from tpp.models.decoders.conditional_poisson import ConditionalPoissonDecoder
from tpp.models.decoders.conditional_poisson_cm import ConditionalPoissonCMDecoder
from tpp.models.decoders.hawkes import HawkesDecoder
from tpp.models.decoders.log_normal_mixture import LogNormalMixtureDecoder
from tpp.models.decoders.mlp_cm import MLPCmDecoder
from tpp.models.decoders.mlp_mc import MLPMCDecoder
from tpp.models.decoders.neural_hawkes import NeuralHawkesDecoder
from tpp.models.decoders.poisson import PoissonDecoder
from tpp.models.decoders.rmtpp import RMTPPDecoder
from tpp.models.decoders.rmtpp_cm import RMTPPCmDecoder
from tpp.models.decoders.self_attention_cm import SelfAttentionCmDecoder
from tpp.models.decoders.self_attention_simple_cm import (
    SelfAttentionCmDecoder as SelfAttentionSimpleCmDecoder,
)
from tpp.models.decoders.self_attention_mc import SelfAttentionMCDecoder

ENCODER_CLASSES = {
    "gru": GRUEncoder,
    "identity": IdentityEncoder,
    "mlp-fixed": MLPFixedEncoder,
    "mlp-variable": MLPVariableEncoder,
    "stub": StubEncoder,
    "selfattention": SelfAttentionEncoder,
}
DECODER_CLASSES = {
    "conditional-poisson": ConditionalPoissonDecoder,
    "conditional-poisson-cm": ConditionalPoissonCMDecoder,
    "hawkes": HawkesDecoder,
    "log-normal-mixture": LogNormalMixtureDecoder,
    "mlp-cm": MLPCmDecoder,
    "mlp-mc": MLPMCDecoder,
    "neural-hawkes": NeuralHawkesDecoder,
    "poisson": PoissonDecoder,
    "rmtpp": RMTPPDecoder,
    "rmtpp-cm": RMTPPCmDecoder,
    "selfattention-cm": SelfAttentionCmDecoder,
    "selfattention-simple-cm": SelfAttentionSimpleCmDecoder,
    "selfattention-mc": SelfAttentionMCDecoder,
}

ENCODER_NAMES = sorted(list(ENCODER_CLASSES.keys()))
DECODER_NAMES = sorted(list(DECODER_CLASSES.keys()))

CLASSES = {"encoder": ENCODER_CLASSES, "decoder": DECODER_CLASSES}
NAMES = {"encoder": ENCODER_NAMES, "decoder": DECODER_NAMES}


def instantiate_encoder_or_decoder(
    args: Dict, marks: int, component="encoder"
) -> nn.Module:
    assert component in {"encoder", "decoder"}

    prefix, classes = component + "_", CLASSES[component]

    kwargs = {k[len(prefix) :]: v for k, v in args.items() if k.startswith(prefix)}
    kwargs["marks"] = marks

    name = args[component]

    if name not in classes:
        raise ValueError(
            "Unknown {} class {}. Must be one of {}.".format(
                component, name, NAMES[component]
            )
        )

    component_class = classes[name]
    component_instance = component_class(**kwargs)

    print("Instantiated {} of type {}".format(component, name))
    print("kwargs:")
    pprint(kwargs)
    print()

    return component_instance


def get_model(
    encoder_decoder_args: Dict,
    marks: int,
    multi_labels: bool,
    include_poisson: bool,
    use_coefficients: bool,
) -> EncDecProcess:
    encoder_decoder_args["decoder_units_mlp"] = encoder_decoder_args[
        "decoder_units_mlp"
    ] + [marks]

    decoder: Decoder
    decoder = instantiate_encoder_or_decoder(
        encoder_decoder_args, marks, component="decoder"
    )

    if decoder.input_size is not None:
        encoder_decoder_args["encoder_units_mlp"] = encoder_decoder_args[
            "encoder_units_mlp"
        ] + [decoder.input_size]

    encoder: Encoder
    encoder = instantiate_encoder_or_decoder(
        encoder_decoder_args, marks, component="encoder"
    )

    process = EncDecProcess(encoder=encoder, decoder=decoder, multi_labels=multi_labels)

    if include_poisson:
        processes = {process.name: process}
        processes.update({"poisson": PoissonProcess(marks=process.marks)})
        process = ModularProcess(processes=processes, use_coefficients=use_coefficients)

    process = process.to(device=device)

    return process
