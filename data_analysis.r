library(tidyverse)
library(tidybayes)
library(lubridate)
library(ggplot2)
library(cowplot)
library(brms)
library(BayesFactor)
library(bayestestR)
library(Bayesrel)
library(GGally)
library(rempsyc)

# rstan crashes on my computer, so I use cmdstanr instead
options(brms.backend = "cmdstanr")

###### Prepare data ######
## All data
data_all <- read_csv("jul_13_full_cleaned.csv")

sum(rowSums(!is.na(data_all[6:146])) > 100)
sum(rowSums(!is.na(data_all[6:146])) > 50) - sum(rowSums(!is.na(data_all[6:146])) > 100)
sum(rowSums(!is.na(data_all[6:146])) < 50)

## Compute how many people completed 1, 2, or 3 versions of the scales
# Create a df to store the data
versions_df <- data.frame(scale = "dummy_scale",
                          version = "dummy_version",
                          n = 0,
                          stringsAsFactors = FALSE)

# Count how many people completed all 3 versions of each scale



# Scale names
scales_qualtrics <- c("FNE", "FOMO", "MW: D&S", "C", "PE") # Scale names on Qualtrics
scales_local <- c("fne", "fomo", "mw-d&s", "neo-pi-c", "barchard-pe") # Scale names on local machine (I was too lazy to change the names on Qualtrics, sorry)

# Create a df to store the data
scale_data_df <- data.frame(scale = "dummy_scale",
                            item_item_comparison = "dummy_comparison",
                            OG_iicm_upper = 0,
                            SP_iicm_upper = 0,
                            LP_iicm_upper = 0,
                            OG_ssm_upper = 0,
                            SP_ssm_upper = 0,
                            LP_ssm_upper = 0,
                            stringsAsFactors = FALSE)

# Loop through each scale
for (scale in 1:length(scales_qualtrics)){
  # Get the item responses for each scale version
  OG <- select(data_all, starts_with(paste0(scales_qualtrics[scale], " (OG)")))
  SP <- select(data_all, starts_with(paste0(scales_qualtrics[scale], " (SP)")))
  LP <- select(data_all, starts_with(paste0(scales_qualtrics[scale], " (LP)")))

  # Compute inter-item correlation matrices (IICMs) (absolute values)
  OG_iicm <- abs(cor(OG, use = "pairwise.complete.obs"))
  SP_iicm <- abs(cor(SP, use = "pairwise.complete.obs"))
  LP_iicm <- abs(cor(LP, use = "pairwise.complete.obs"))

  # Get upper triangles of IICMs
  OG_iicm_upper <- OG_iicm[upper.tri(OG_iicm)]
  SP_iicm_upper <- SP_iicm[upper.tri(SP_iicm)]
  LP_iicm_upper <- LP_iicm[upper.tri(LP_iicm)]

  # Read in the semantic similarity matrices (SSMs)
  OG_ssm  <- read_csv(paste0("ssms/", scales_local[scale], "_OG_ssm.csv"), col_names = FALSE)
  SP_ssm  <- read_csv(paste0("ssms/", scales_local[scale], "_SP_ssm.csv"), col_names = FALSE)
  LP_ssm  <- read_csv(paste0("ssms/", scales_local[scale], "_LP_ssm.csv"), col_names = FALSE)

  # For the FOMO scale, remove the first item's semantic similarity scores (see FOMO cleaning intermission above)
  if (scales_local[scale] == "fomo"){
    OG_ssm <- OG_ssm[,-1][-1,]
    SP_ssm <- SP_ssm[,-1][-1,]
    LP_ssm <- LP_ssm[,-1][-1,]
  }

  # Get upper triangles of SSMs
  OG_ssm_upper <- OG_ssm[upper.tri(OG_ssm)]
  SP_ssm_upper <- SP_ssm[upper.tri(SP_ssm)]
  LP_ssm_upper <- LP_ssm[upper.tri(LP_ssm)]

  # Create a vector of all the item-item comparisons
  comparisons <- character(length(OG_ssm_upper))
  counter <- 1
  for (item1 in 2:(dim(OG_ssm)[1])){
    for (item2 in 1:(item1 - 1)){
      if (scales_local[scale] == "fomo"){
        comparisons[counter] <- paste0(item1 + 1, "-", item2 + 1)
      } else {
        comparisons[counter] <- paste0(item1, "-", item2)
      }
      counter <- counter + 1
    }
  }

  # Create dataframe with all the data as separate columns + append to scale_data_df
  data <- data.frame(scale = scales_local[scale], item_item_comparison = comparisons, OG_iicm_upper, SP_iicm_upper, LP_iicm_upper, OG_ssm_upper, SP_ssm_upper, LP_ssm_upper)
  scale_data_df <- rbind(scale_data_df, data)
}

# Remove the first row (dummy row)
scale_data_df <- scale_data_df[-1,]

# Write scale_data_df
write_csv(scale_data_df, "scale_data_df.csv")

###### End of prepare data ######


###### Descriptives ######
desc <- scale_data_df %>%
  # group_by(scale) %>%
  summarise(mean_OG_iicm = mean(OG_iicm_upper),
            mean_SP_iicm = mean(SP_iicm_upper),
            mean_LP_iicm = mean(LP_iicm_upper),
            sd_OG_iicm = sd(OG_iicm_upper),
            sd_SP_iicm = sd(SP_iicm_upper),
            sd_LP_iicm = sd(LP_iicm_upper),
            mean_OG_ssm = mean(OG_ssm_upper),
            mean_SP_ssm = mean(SP_ssm_upper),
            mean_LP_ssm = mean(LP_ssm_upper),
            sd_OG_ssm = sd(OG_ssm_upper),
            sd_SP_ssm = sd(SP_ssm_upper),
            sd_LP_ssm = sd(LP_ssm_upper))

# H1
scale_iicm_ssm_corrs <-scale_data_df %>%
  # group_by(scale) %>%
  summarise(cor_OG = cor(OG_iicm_upper, OG_ssm_upper),
            cor_SP = cor(SP_iicm_upper, SP_ssm_upper),
            cor_LP = cor(LP_iicm_upper, LP_ssm_upper))

scale_between_version_rels <- scale_data_df %>%
  # group_by(scale) %>%
  summarise(cor_OG_SP_bvc = cor(OG_iicm_upper, SP_iicm_upper),
            cor_OG_LP_bvc = cor(OG_iicm_upper, LP_iicm_upper),
            cor_SP_LP_bvc = cor(SP_iicm_upper, LP_iicm_upper),
            cor_OG_SP_bvssm = cor(OG_ssm_upper, SP_ssm_upper),
            cor_OG_LP_bvssm = cor(OG_ssm_upper, LP_ssm_upper),
            cor_SP_LP_bvssm = cor(SP_ssm_upper, LP_ssm_upper))

# Plot correlations between IICM columns, grouped by scale, and colored by scale version (i.e., OG, SP, LP)
scale_data_df %>%
  group_by(scale) %>%
  summarise(cor_OG_SP = cor(OG_iicm_upper, SP_iicm_upper),
            cor_OG_LP = cor(OG_iicm_upper, LP_iicm_upper),
            cor_SP_LP = cor(SP_iicm_upper, LP_iicm_upper)) %>%
  pivot_longer(cols = c(cor_OG_SP, cor_OG_LP, cor_SP_LP),
               names_to = "comparison",
               values_to = "cor") %>%
  ggplot(aes(x = scale, y = cor, color = comparison)) +
  geom_point() +
  theme_bw() +
  theme(legend.position = "none")



apply(scale_iicm_ssm_corrs[2:4], 2, mean)



desc_nice <- nice_table(desc)
flextable::save_as_docx(desc_nice, path = "nice_tablehere.docx")

scale_iicm_ssm_corrs_nice <- nice_table(scale_iicm_ssm_corrs)
flextable::save_as_docx(scale_iicm_ssm_corrs_nice, path = "scale_iicm_ssm_corrs_nice.docx")

scale_between_version_rels_nice <- nice_table(scale_between_version_rels)
flextable::save_as_docx(scale_between_version_rels_nice, path = "scale_between_version_rels_nice.docx")

###### End of descriptive statistics ######


###### Bayesian analysis ######
### Bayesian regressions with weakly informative priors
## beta priors: student_t(3, 0, 2.5)
## sigma priors: half_cauchy(2.5)
scale_data_df <- read_csv("scale_data_df.csv")

priors <- c(set_prior("student_t(3, 0, 2.5)", class = "b"),
            set_prior("cauchy(0, 2.5)", class = "sigma"))

## H2
# "... the original and strict-paraphrase semantic similarities will be equally good predictors of
# the original scale and strict-paraphrase scale item-item relationships, respectively." (p. 7)

## H3
# "... whereas the loose-paraphrase similarities will be poorer predictors of the loose-paraphrase item relationships." (p. 7)

# Regress OG IICMs on SSMs
# scaled_scale_data_df = scale(scale_data_df) # scaling makes no numerical difference in any of the analyses

OG_fit <- brm(OG_iicm_upper ~ OG_ssm_upper,
              data = scale_data_df,
              prior = priors,
              chains = 4,
              iter = 4000,
              cores = 4,
              seed = 123)

#~~~~ Random intercepts
library(lme4)
random_icept = lmer(OG_iicm_upper ~ OG_ssm_upper + (1| scale), data = scale_data_df)
summary(random_icept)

predicted_values<- modelr::data_grid(scale_data_df, OG_ssm_upper, scale) %>% 
                   modelr::add_predictions(random_icept)


plotplot <- predicted_values %>% 
  ggplot(aes(OG_ssm_upper, pred, color = scale))+
  geom_line()+
  geom_point(data = scale_data_df, aes(OG_ssm_upper, OG_iicm_upper, color = scale))
#~~~~

# Regress SP IICMs on SSMs
SP_fit <- brm(SP_iicm_upper ~ SP_ssm_upper,
              data = scale_data_df,
              prior = priors,
              chains = 4,
              iter = 4000,
              cores = 4,
              seed = 123)

# Regress LP IICMs on SSMs
LP_fit <- brm(LP_iicm_upper ~ LP_ssm_upper,
              data = scale_data_df,
              prior = priors,
              chains = 4,
              iter = 4000,
              cores = 4,
              seed = 123)


# Get posterior regression coefficient samples
b_ssm_posterior_samples <- data.frame(OG_b_ssm = posterior_samples(OG_fit)$b_OG_ssm_upper,
                                      SP_b_ssm = posterior_samples(SP_fit)$b_SP_ssm_upper,
                                      LP_b_ssm = posterior_samples(LP_fit)$b_LP_ssm_upper)

# Compute OG & SP difference distribution
H2 <- b_ssm_OG_SP_delta <- b_ssm_posterior_samples$OG_b_ssm - b_ssm_posterior_samples$SP_b_ssm

# Compute OG & LP difference distribution
H3 <- b_ssm_OG_LP_delta <- b_ssm_posterior_samples$OG_b_ssm - b_ssm_posterior_samples$LP_b_ssm

# SP & LP just for Table
b_ssm_SP_LP_delta <- b_ssm_posterior_samples$SP_b_ssm - b_ssm_posterior_samples$LP_b_ssm
hdi(b_ssm_SP_LP_delta, prob = 0.95)

plot(density(H2))


# Compute 95% HDIs
hdi(H2, prob = 0.95)
hdi(H3, prob = 0.95)

### Post-hoc, plotting, etc.
# Posterior predictive checks
pp_check(OG_fit, ndraws = 10)
pp_check(SP_fit, ndraws = 10)
pp_check(LP_fit, ndraws = 10)

## Internal consistency
# Compute Cronbach's alpha for each scale and version
scale_names <- unique(gsub("_.*", "", names(data_all[6:146])))
rel_alpha_cis <- list()
rel_omega_cis <- list()
for (scale in scale_names){
  print(scale)
  scale_data <- data_all[grep(scale, names(data_all), fixed = TRUE)]
  rel_alpha_cis[[scale]] <- strel(scale_data, "alpha", freq = FALSE)
  rel_omega_cis[[scale]] <- strel(scale_data, "omega", freq = FALSE)
}

## Alternate-form reliability (AFR)
# Create dataframe to store sum scores
sum_scores <- data.frame(matrix(NA, nrow = nrow(data_all), ncol = length(scale_names)))
colnames(sum_scores) <- scale_names

# Compute sum scores per scale, per version
for (scale in scale_names)
  sum_scores[scale] <- rowSums(data_all[grep(scale, names(data_all), fixed = TRUE)], na.rm = TRUE)
  sum_scores[sum_scores == 0] <- NA

# Compute AFRs (i.e., correlations between same-scale sum scores)
# create df to score sum scores, with version as column name
plot_list <- list()
triplet_cor_list <- list()

# Get sum scores for each version of the scale, then plot them against each other
for (scale_all in scales_qualtrics){
  triplet <- sum_scores[grep(scale_all, names(sum_scores))]
  triplet_plot <- ggpairs(triplet,
                          title = paste0(scale_all, ": Alternative-form reliability"),
                          upper = list(continuous = wrap("cor", size = 3)),
                          columnLabels = c("Original items", "Strict-par. items", "Loosely-par. items"))

  # Store plot
  plot_list[[scale_all]] <- triplet_plot

  # Store correlation matrix
  triplet_cor <- cor(triplet, use = "pairwise.complete.obs")
  triplet_cor_list[[scale_all]] <- triplet_cor
}

## Plotting regression models
xmin <- min(scale_data_df[6:8])
xmax <- 1
ymin <- 0 
ymax <- max(scale_data_df[3:5])

# OG regression model
OG_plot <- scale_data_df %>%
    add_predicted_draws(OG_fit, seed = 123) %>%  # adding the posterior distribution
    ggplot(aes(x = OG_ssm_upper, y = OG_iicm_upper, color = scale)) +  
    stat_lineribbon(aes(y = .prediction), .width = c(.95, .80),  # regression line and CI
                    alpha = 0.5, colour = "black") +
    geom_point(data = scale_data_df, size = 3) +   # raw data
    scale_fill_brewer(palette = "Greys") +
    # geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    ggtitle("OG Regression") + 
    # xlab("\nItem-item semantic similarity") +
    # ylab("Inter-item correlation\n") + 
    xlim(xmin, xmax) + 
    ylim(ymin, ymax) +
    theme_bw() +
    theme(legend.title = element_blank(),
          legend.position = c(0.15, 0.85),
          plot.title = element_text(size = 17),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.x = element_text(size = 13),
          axis.text.y = element_text(size = 13))

# SP regression model
SP_plot <- scale_data_df %>%
    add_predicted_draws(SP_fit, seed = 123) %>%  # adding the posterior distribution
    ggplot(aes(x = SP_ssm_upper, y = SP_iicm_upper, color = scale)) +  
    stat_lineribbon(aes(y = .prediction), .width = c(.95, .80),  # regression line and CI
                    alpha = 0.5, colour = "black") +
    geom_point(data = scale_data_df, size = 3) +   # raw data
    scale_fill_brewer(palette = "Greys") +
    # geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    ggtitle("SP Regression") + 
    # ylab("Inter-item correlation\n") + 
    # xlab("\nItem-item semantic similarity") +
    xlim(xmin, xmax) + 
    ylim(ymin, ymax) +
    theme_bw() +
    theme(legend.title = element_blank(),
          legend.position = c(0.15, 0.85),
          plot.title = element_text(size = 17),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.x = element_text(size = 13),
          axis.text.y = element_text(size = 13))

# LP regression model
LP_plot <- scale_data_df %>%
    add_predicted_draws(LP_fit, seed = 123) %>%  # adding the posterior distribution
    ggplot(aes(x = LP_ssm_upper, y = LP_iicm_upper, color = scale)) +  
    stat_lineribbon(aes(y = .prediction), .width = c(.95, .80),  # regression line and CI
                    alpha = 0.5, colour = "black") +
    geom_point(data = scale_data_df, size = 3) +   # raw data
    scale_fill_brewer(palette = "Greys") +
    # geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    ggtitle("LP Regression") + 
    # ylab("Inter-item correlation\n") +
    # xlab("\nItem-item semantic similarity") +
    xlim(xmin, xmax) + 
    ylim(ymin, ymax) +
    theme_bw() +
    theme(legend.title = element_blank(),
          legend.position = c(0.15, 0.85),
          plot.title = element_text(size = 17),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.x = element_text(size = 13),
          axis.text.y = element_text(size = 13))

reg_together <- ggarrange(OG_plot, SP_plot, LP_plot,
                          ncol = 2, nrow = 2,
                          common.legend = TRUE)
annotate_figure(reg_together,
                left = text_grob("Inter-item correlation", size = 17, rot = 90),
                bottom = text_grob("Inter-item semantic similarity", size = 17)
                )

plot_grid(OG_plot, SP_plot, LP_plot + rremove("x.text"), 
          labels = c("A", "B", "C"),
          ncol = 2, nrow = 2)