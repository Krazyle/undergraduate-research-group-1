#import "title-page.typ": title-page

#let single = 1.15em
#let double = 2 * single
#let line-gap = double - 1em
#let cell-pad = line-gap / 2

#let ceu-format(title: none, authors: (), body) = {
  set document(title: title, author: authors)

  set page(
    paper: "us-letter",
    margin: (left: 1.5in, right: 1in, top: 1in, bottom: 1in),
    header: context align(right, counter(page).display("1")),
    numbering: none,
  )

  set text(font: "Times New Roman", size: 12pt, lang: "en", region: "US")

  set text(top-edge: 1em, bottom-edge: 0em)
  set par(
    leading: line-gap,
    spacing: line-gap,
    justify: true,
    first-line-indent: (amount: 0.5in, all: true),
  )
  set list(spacing: line-gap)
  set enum(spacing: line-gap)

  set heading(numbering: "1.1")
  show heading: set text(size: 12pt, weight: "bold")
  show heading: set block(above: line-gap, below: line-gap)

  show heading.where(body: []): set heading(numbering: none)
  show heading.where(body: []): none

  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    counter(figure.where(kind: "quarto-float-fig")).update(0)
    counter(figure.where(kind: "quarto-float-tbl")).update(0)
    set align(center)
    if it.numbering != none [
      Chapter #counter(heading).display("1") \
      #it.body
    ] else {
      it.body
    }
  }

  set figure(numbering: n => numbering("1.1", counter(heading).get().first(), n))
  set figure(gap: line-gap)
  show figure: set block(above: line-gap, below: line-gap)
  show figure.where(kind: "quarto-float-tbl"): set block(breakable: true, above: line-gap - cell-pad, below: line-gap - cell-pad)
  show figure.where(kind: "quarto-float-tbl"): set figure(gap: line-gap - cell-pad)
  show figure.caption: it => align(left, block(sticky: true)[
    #strong[#it.supplement #context it.counter.display(it.numbering)] \
    #it.body
  ])

  set table(stroke: 0.5pt, inset: (x: 0.5em, y: cell-pad))
  set table.hline(stroke: 0.5pt)
  set table.header(repeat: false)
  show table.cell.where(y: 0): set text(weight: "bold")
  show table: set par(justify: false, first-line-indent: 0in)

  show math.equation.where(block: true): set block(above: line-gap, below: line-gap)

  show <refs>: set par(first-line-indent: 0in, hanging-indent: 0.5in)

  title-page(title: title, authors: authors)
  body
}
