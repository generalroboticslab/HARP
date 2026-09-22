# HARP rover

CAD for one fly-drive rover module.

![HARP rover CAD preview](images/rover_front_pitch.png)

## Open the assembly

Set the Creo working directory to `cad/`, then open [assemble.asm.53](cad/assemble.asm.53).
Keep the files in that directory together and retain their original names.
The assembly header identifies Creo 12.4.0.0.

## Files

- `cad/`: one assembly and 15 unique component part files.
- `images/`: the supplied CAD preview.
- [inventory.csv](inventory.csv): source paths, quantities, save numbers, sizes, and SHA-256 checksums.

## Component inventory

The saved assembly member table contains 34 part instances. Quantities below
are for one module and count CAD instances.

| Selected part file | Quantity |
| --- | ---: |
| [beam.prt.9](cad/beam.prt.9) | 4 |
| [drone-fly-motor.prt.4](cad/drone-fly-motor.prt.4) | 4 |
| [drone_frame.prt.4](cad/drone_frame.prt.4) | 1 |
| [guiding_wheel.prt.5](cad/guiding_wheel.prt.5) | 6 |
| [motor_base.prt.10](cad/motor_base.prt.10) | 4 |
| [pin.prt.6](cad/pin.prt.6) | 2 |
| [prt0006.prt.1](cad/prt0006.prt.1) | 1 |
| [prt0008.prt.1](cad/prt0008.prt.1) | 1 |
| [prt0009.prt.1](cad/prt0009.prt.1) | 1 |
| [prt0010.prt.1](cad/prt0010.prt.1) | 1 |
| [sprocket.prt.8](cad/sprocket.prt.8) | 2 |
| [stand0804_2.prt.3](cad/stand0804_2.prt.3) | 2 |
| [top_v2.prt.6](cad/top_v2.prt.6) | 1 |
| [track-in.prt.6](cad/track-in.prt.6) | 2 |
| [track_motor.prt.4](cad/track_motor.prt.4) | 2 |

## Version selection and validation

Each model uses the highest numbered save available in the source `video/CAD/`
directory. The extracted references identify model names, but do not establish
the numbered saves originally loaded in `whole-system-paper.asm.8`.

All named assembly members are included. There are no nested assemblies in
this saved member table. The selected part reference records were inspected
and yielded no additional component model dependencies. Copy checksums were
verified against the source files. Opening and regenerating the assembly in
Creo remains to be checked.

Original filenames, including generic `prt000*` names and spelling, are retained.
The preview is copied from the source folder; its exact CAD save is unverified.
