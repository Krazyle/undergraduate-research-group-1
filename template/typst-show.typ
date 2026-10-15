#import "template/ceu-format.typ": ceu-format

#show: ceu-format.with(
  title: [$title$],
  authors: ($for(by-author)$"$it.name.literal$", $endfor$),
)
