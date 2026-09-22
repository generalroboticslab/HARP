# HARP drilling payload

CAD for one flying payload module with its drilling mechanism.

![HARP drilling payload CAD preview](images/drill_drone_front_pitch.png)

## Open the assembly

Set the Creo working directory to `cad/`, then open [drill_drone.asm.20](cad/drill_drone.asm.20).
Keep the files in that directory together and retain their original names.
The assembly header identifies Creo 12.4.0.0.

## Files

- `cad/`: one assembly and 16 unique component part files.
- `images/`: the supplied CAD preview.
- [inventory.csv](inventory.csv): source paths, quantities, save numbers, sizes, and SHA-256 checksums.
- `exports/stl/`: two existing part exports.

## Component inventory

The saved assembly member table contains 35 part instances. Quantities below
are for one module and count CAD instances.

| Selected part file | Quantity |
| --- | ---: |
| [10x4_5l.prt.1](cad/10x4_5l.prt.1) | 1 |
| [10x4_5r.prt.1](cad/10x4_5r.prt.1) | 1 |
| [beam275.prt.3](cad/beam275.prt.3) | 4 |
| [beam_l20_500mm.prt.4](cad/beam_l20_500mm.prt.4) | 4 |
| [dirll_frame_body.prt.46](cad/dirll_frame_body.prt.46) | 1 |
| [drill_drone_frame_topper.prt.3](cad/drill_drone_frame_topper.prt.3) | 2 |
| [drill_feed.prt.4](cad/drill_feed.prt.4) | 1 |
| [drill_frame_beam_holder.prt.3](cad/drill_frame_beam_holder.prt.3) | 4 |
| [drill_head.prt.2](cad/drill_head.prt.2) | 1 |
| [drill_motor_board_use.prt.13](cad/drill_motor_board_use.prt.13) | 1 |
| [drill_nut.prt.5](cad/drill_nut.prt.5) | 1 |
| [drone-fly-motor.prt.4](cad/drone-fly-motor.prt.4) | 4 |
| [magnet_son_holder.prt.24](cad/magnet_son_holder.prt.24) | 4 |
| [motor_base-20.prt.1](cad/motor_base-20.prt.1) | 4 |
| [prt0004.prt.1](cad/prt0004.prt.1) | 1 |
| [prt0005.prt.4](cad/prt0005.prt.4) | 1 |

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

## Existing STL exports

These files share names with included native parts. Their export revisions
and geometry match to the selected native saves have not been verified.
Check them against the native parts before fabrication.

- [drill_motor_board_use.stl](exports/stl/drill_motor_board_use.stl)
- [magnet_son_holder.stl](exports/stl/magnet_son_holder.stl)
