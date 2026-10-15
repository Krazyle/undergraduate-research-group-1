#import "figure-style.typ": *
#show: figure-style

#ipo-diagram(
  height: 2.9in,
  [
    A Taglish disaster tweet

    Example \
    "Kailangan namin ng tubig at pagkain dito sa Brgy. San Isidro, Rodriguez. Wala pang relief na dumarating."
  ],
  [
    + Split the tweet into tokens
    + The shared encoder reads the tokens and converts them into numbers
    + Two task heads make predictions from the same numbers
      - Classification head
      - Named entity recognition head
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
