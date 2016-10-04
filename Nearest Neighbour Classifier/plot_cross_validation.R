library(ggplot2)

setwd("~/Documents/Statistical-Machine-Learning-Project/Nearest Neighbour Classifier")

data = read.csv("nearestNeighbour_CV.csv", header = TRUE, sep = ",")
data$p = factor(data$p, levels = c(1, 2, 3), labels = c("Manhattan Distance", "Euclidean Distance", "L3-Norm"))

# line plot of cost for each schedule varying number of customers
ggsave(filename = "~/Documents/Statistical-Machine-Learning-Project/Nearest Neighbour Classifier/Nearest_Neighbour_CV.eps",
       width = 20, height = 10,
       plot =
         ggplot(data = data, aes(x = factor(n_neighbors), y = cross_validation_accuracy, group = factor(p),
                                 colour = factor(p))) +
         stat_summary(fun.y = "mean", geom = "point", size = 3) +
         stat_summary(fun.y = "mean", geom = 'line', size = 1.5) +
         labs(x = "Number of Neighbours (k)", y = "Cross Validation Accuracy") +
         guides(colour = guide_legend(override.aes = list(shape = 15, size = 10))) + 
         coord_cartesian(ylim = c(0.48, 0.68)) +
         scale_y_continuous(breaks = seq(from = 0.48, to = 0.68, by = 0.04)) +
         theme_bw() +
         theme(axis.text.x = element_text(size = 24), axis.text.y = element_text(size = 24),
               axis.title.x = element_text(face = "bold", size = 28, margin = margin(20, 0, 20, 0)),
               axis.title.y = element_text(face = "bold", size = 28, margin = margin(0, 20, 0, 20)),
               legend.title = element_blank(), legend.text = element_text(size = 24),
               legend.key = element_blank(), legend.position = 'top')
)