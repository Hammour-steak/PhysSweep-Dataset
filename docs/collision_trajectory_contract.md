# Collision and trajectory publication integrity

Generic resolved collision proxies describe the collision profile actually passed
to PyBullet. Compound profiles preserve every child shape, size, local position
and rotation. The geometry envelope remains available for visual placement and
CCD sizing; it is not a replacement for compound collision geometry.

Publication accepts the historical frozen representation only when a compound
profile's resolved proxy exactly equals its source geometry envelope. It recovers
the complete proxy from the hash-verified source metadata without modifying the
frozen source, resolved scene or trajectory. Other disagreements are errors.
New simulations require the resolved proxy to match the executed source profile.

Canonical trajectory build, base verification, sweep verification and resume
share numerical integrity validation: array dimensions, finite values, object
order, increasing time from zero, metadata frame rate, unit quaternions with
continuous signs, integer contact counts and metadata initial states. Contact
counts are checked before int32 conversion. These checks do not enforce motion
outcome rules or restrict the magnitude of sweep perturbations.
