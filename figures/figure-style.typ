#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#let figure-style(body) = {
  set page(width: auto, height: auto, margin: 2pt)
  set text(font: "Times New Roman", size: 10pt, hyphenate: false)
  set par(justify: false)
  set list(spacing: 0.8em)
  set enum(spacing: 0.8em)
  body
}

#let box-style = (inset: 8pt, stroke: 0.9pt, corner-radius: 5pt)

#let group-style = (inset: 4pt, stroke: (thickness: 0.6pt, dash: "dashed"), corner-radius: 8pt)

#let ipo-diagram(height: auto, input, process, output) = diagram(
  spacing: (0.25in, 0.15in),
  edge-stroke: 1.2pt,
  ..range(3).map(x => node((x, 0), strong(("INPUT", "PROCESS", "OUTPUT").at(x)))),
  ..(input, process, output).enumerate().map(((x, body)) => node(
    (x, 1),
    align(left, body),
    width: 1.8in,
    height: height,
    ..box-style,
  )),
  edge((0, 1), (1, 1), "-|>"),
  edge((1, 1), (2, 1), "-|>"),
)
