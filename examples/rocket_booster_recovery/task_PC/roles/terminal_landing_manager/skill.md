# Terminal landing manager

Own the last roughly 50 m: height-speed corridor, upright freeze, gentle
throttle shaping, and a deterministic hybrid approach/contact-expected state
machine. Keep powered closed-loop control active through first leg contact.
Never create an engine-cut or low-thrust gear-sink/drop phase whose safety
depends on suspension damping. The evaluator stops scoring at interpolated
first contact; post-contact dwell/bounce/load/slip is unscored and must not be
used as a surrogate objective. Do not alter contact geometry or termination.
