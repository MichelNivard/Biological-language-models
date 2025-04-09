library(ggplot2)
library(ggforce)

# Define positions of elements
components <- data.frame(
  label = c("Sequence", "Image", "Q", "K", "V"),
  x = c(1, 1, 3, 4, 5),
  y = c(4, 2, 4, 2, 2),
  color = c("lightgreen", "lightblue", "yellow", "skyblue", "salmon")
)

# Base plot
ggplot(components) +
  geom_rect(aes(xmin = x - 0.4, xmax = x + 0.4, ymin = y - 0.3, ymax = y + 0.3, fill = label), color = "black") +
  geom_text(aes(x = x, y = y, label = label), size = 5) +
  scale_fill_manual(values = setNames(components$color, components$label)) +
  xlim(0, 6) + ylim(1, 5) +
  theme_void() +
  theme(legend.position = "none")
