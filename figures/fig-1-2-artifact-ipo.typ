#import "figure-style.typ": *
#show: figure-style

#ipo-diagram(
  height: 2.3in,
  [
    A Taglish disaster post

    "Kailangan namin ng tubig at pagkain dito sa Brgy. San Isidro, Rodriguez. Wala pang relief na dumarating."
  ],
  [
    + Split the post into tokens
    + The pretrained encoder, shared by both tasks, reads the tokens and converts them into numbers
    + Two task heads, one for each task, make predictions from the same numbers
  ],
  [
    - Humanitarian category \
      Requests or urgent needs
    - Named entities \
      Brgy. San Isidro (location) \
      Rodriguez (location) \
      tubig, pagkain (resource)
  ],
)
