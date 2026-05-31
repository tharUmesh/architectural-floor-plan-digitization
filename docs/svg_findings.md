# SVG Structure Findings
**Date:** 2026-05-31
**Dataset:** CubiCasa5k
**Audited by:** Tharun Umesh

## Namespace Declarations
`http://www.w3.org/2000/svg`

## How Element Types Are Encoded
The `class` attribute on `<g>` elements encodes the type.
Some structural elements also have a matching `id` attribute (e.g. id="Door").

## Tag Types Used for Geometry
All elements use `<polygon>` with a `points` attribute.
Points are space-separated or comma-separated x,y coordinate pairs.

## Coordinate System — TWO TYPES
Structural (Wall, Door, Window):
  Polygon points are absolute SVG pixel coordinates. No transform needed.

Furniture (Toilet, Sink, Staircase):
  Polygon points are LOCAL (start at 0,0).
  Parent <g> has transform="matrix(a,b,c,d,e,f)" that must be applied.
  Formula: real_x = a*x + c*y + e  |  real_y = b*x + d*y + f

## Exact Class Name Strings
| Our Class | SVG class attribute value        |
|-----------|----------------------------------|
| Door      | "Door Swing Beside", "Door ..."  |
| Window    | "Window Regular", "Window ..."   |
| Wall      | "Wall", "Wall External"          |
| Staircase | "FixedFurniture Staircase"        |
| Toilet    | "FixedFurniture Toilet"          |
| Sink      | "FixedFurniture Sink"            |

## Class Imbalance — Action Required in Phase 5

Annotation counts across 4,967 folders:
- Wall:      131,150  (53.7%)
- Door:       49,833  (20.4%)
- Window:     43,956  (18.0%)
- Sink:        7,844   (3.2%)
- Toilet:      6,949   (2.8%)
- Staircase:   4,629   (1.9%) 

Imbalance ratio Wall:Toilet ≈ 19:1

Phase 5 mitigation strategies to apply:
1. YOLOv8's built-in class weighting via the `cls` loss weight
2. Mosaic augmentation naturally overrepresents rare-class images
   when those images are sampled more frequently
3. Monitor per-class recall during training — if Toilet/Sink recall
   drops below 0.50, apply targeted augmentation

## Staircase Class Name Resolution

The keyword "Staircase" does not exist in the dataset.
Actual SVG class names for staircase elements:
- class="Stairs" id="Stairs"  → the complete staircase unit (4,629 occurrences)
                                  THIS is the element we annotate
- class="Steps"  id="Steps"   → individual step lines within a staircase (8,100)
                                  These are CHILDREN of Stairs, like Panel/Threshold
                                  are children of Door. DO NOT annotate separately.

Fix applied: configs/dataset.yaml updated to svg_class_tags.Staircase: ["Stairs"]

## Coordinate Type for Stairs (To Verify in Phase 2)
Stairs elements use id="Stairs" similar to how Door uses id="Door".
Likely uses direct polygon coordinates (like Wall/Door/Window), NOT the
furniture-style matrix transform pattern (like Toilet/Sink).
Must verify in Phase 2 by inspecting an actual Stairs polygon.

## Surprises or Edge Cases
Anything unexpected (missing classes, unusual structures, corrupt files): None found in the audited sample.

## Action Required in configs/dataset.yaml
[List the changes needed to svg_class_tags based on findings]