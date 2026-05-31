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

## Surprises or Edge Cases
Anything unexpected (missing classes, unusual structures, corrupt files): None found in the audited sample.

## Action Required in configs/dataset.yaml
[List the changes needed to svg_class_tags based on findings]