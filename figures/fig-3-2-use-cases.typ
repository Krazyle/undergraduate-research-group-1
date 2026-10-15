#import "figure-style.typ": *
#show: figure-style

#let stick-figure = box(width: 0.3in, height: 0.42in, {
  let s = 0.9pt
  place(dx: 0.1in, dy: 0in, circle(radius: 0.05in, stroke: s))
  place(line(start: (0.15in, 0.1in), end: (0.15in, 0.26in), stroke: s))
  place(line(start: (0.04in, 0.16in), end: (0.26in, 0.16in), stroke: s))
  place(line(start: (0.15in, 0.26in), end: (0.06in, 0.42in), stroke: s))
  place(line(start: (0.15in, 0.26in), end: (0.24in, 0.42in), stroke: s))
})

#let actor(pos, name) = {
  let (x, y) = pos
  (node(pos, stick-figure, inset: 0pt), node((x, y + 0.8), strong(name), height: 0pt))
}

#let hand(pos, side) = (rel: (side * 0.11in, -0.05in), to: pos)

#let use-case-width = 1.9in

#let left-tip(pos) = (rel: (-use-case-width / 2, 0pt), to: pos)
#let right-tip(pos) = (rel: (use-case-width / 2, 0pt), to: pos)

#let use-case(pos, body) = node(
  pos,
  body,
  shape: fletcher.shapes.ellipse,
  width: use-case-width,
  height: 0.38in,
  stroke: 0.8pt,
  fill: luma(96%),
)

#let staff-cases = (
  [Import posts],
  [Analyze posts],
  [View analyzed posts],
  [Correct a label],
  [Filter posts],
  [Mark post status],
  [View summary],
  [Export report],
)

#let researcher-cases = (
  [Load trained model],
  [Export corrected labels],
)

#diagram(
  spacing: (1.1in, 0.17in),
  edge-stroke: 0.8pt,

  node((1, -1), strong[Prototype Web Application]),
  ..(staff-cases + researcher-cases).enumerate().map(((y, body)) => use-case((1, y), body)),
  node(enclose: ((1, -1), (1, 9)), inset: 10pt, stroke: 1pt, fill: luma(98%), snap: false),

  ..actor((0, 3.5))[DRRMO \ Staff],
  ..actor((2, 0))[X],
  ..actor((2, 8.5))[Researcher],

  ..range(staff-cases.len()).map(y => edge(hand((0, 3.5), 1), left-tip((1, y)), snap-to: none)),
  edge(hand((2, 0), -1), right-tip((1, 0)), snap-to: none),
  edge(hand((2, 8.5), -1), right-tip((1, 8)), snap-to: none),
  edge(hand((2, 8.5), -1), right-tip((1, 9)), snap-to: none),
)
