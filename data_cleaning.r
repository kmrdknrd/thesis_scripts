library(tidyverse)
library(usefun)

## Read in the shuffled data
# Data from participants who filled in all three parts
compl_resp <- read_csv2("FULL/jul_11_compl_resp.csv")[-c(1,2),]

# Data from participants who missed at least one part
incompl_resp <- read_csv("FULL/jul_11_incompl_resp.csv")[-c(1,2),]

# Combine the two datasets
combined <- bind_rows(compl_resp, incompl_resp)


## Clean the data
# Remove Status, IPAddress, Progress, ResponseId, RecipientLastName, RecipientFirstName, RecipientEmail, ExternalReference, LocationLatitude, LocationLongitude, DistributionChannel, UserLanguage, tid, userid
combined <- combined %>%
    select(-c(Status, IPAddress, Progress, ResponseId, RecipientLastName, RecipientFirstName, RecipientEmail, ExternalReference, LocationLatitude, LocationLongitude, DistributionChannel, UserLanguage, tid, userid))

## Write the depersonalized data to a csv file
write.csv(combined, "raw_data.csv", row.names = FALSE)

# Remove rows that failed attention checks (`FNE (LP)_9` != 5 & `MW: D&S (SP)_9` != 1), unless they have a value of NA,
# then remove `FNE (LP)_9` and `MW: D&S (SP)_9`,
# then rename `FNE (LP)_10`, `..._11` and `..._12` to `..._9`, `..._10`, and `..._11`, respectively (their values were shifted up by one because of the attention check)
combined <- combined %>%
    filter((`FNE (LP)_9` == 5 | is.na(`FNE (LP)_9`)) & (`MW: D&S (SP)_9` == 1 | is.na(`MW: D&S (SP)_9`))) %>%
    select(-c("FNE (LP)_9", "MW: D&S (SP)_9")) %>%
    rename("FNE (LP)_9" = "FNE (LP)_10",
           "FNE (LP)_10" = "FNE (LP)_11",
           "FNE (LP)_11" = "FNE (LP)_12")


# Keep rows that have Q34 value of 3 and Q37 value of 4 (i.e. English fluency items), then remove Q34 and Q37
combined <- combined %>%
  filter(Q34 == 3 & Q37 == 4) %>%
  select(-c(Q34, Q37))

# Remove rows that have BlockOrder value of NA (i.e., they exited the survey before block randomization),
# then separate BlockOrder into three columns, one for each block
combined <- combined %>%
  filter(!is.na(BlockOrder)) %>%
  separate(BlockOrder, c("Block1", "Block2", "Block3"), sep = c(1, 2))

# Remove rows that have only NAs in columns 6:146 (i.e., they didn't answer any of the questions)
combined <- combined %>%
  filter(!rowSums(is.na(combined[6:146])) == 141)


### FOMO cleaning intermission ###
# In each version of the FOMO scale, an item got duplicated and overwrote an exisiting item:
# - FOMO (OG)_11 overwrote FOMO (OG)_8; they became `FOMO (OG)_11...41` and `FOMO (OG)_11...38`, respectively.
# - FOMO (SP)_6 overwrote FOMO (SP)_1; they became `FOMO (SP)_6...113` and `FOMO (SP)_6...108`, respectively. 
# - FOMO (LP)_6 overwrote FOMO (LP)_1; they became `FOMO (LP)_6...135` and `FOMO (LP)_6...130`.

# "Luckily", FOMO (OG)_8 was an attention check; however, this still resulted in incomplete data for item 1 (missing in SP and LP).

# Therefore, we:
# a) remove the duplicated responses (`FOMO (OG)_11...38`, `FOMO (SP)_6...108`, and `FOMO (LP)_6...130`);
# b) remove the item that can't be compared across scale versions (`FOMO (OG)_1`);
# c) rename `FOMO (OG)_11...41`, `FOMO (SP)_6...113` and `FOMO (LP)_6...135` "back" to `_11`, `_6`, and `_6`, respectively.
# d) rename `FOMO (OG)_9`, `_10` and `_11` to `_8`, `_9` and `_10`, respectively; (their values were shifted up by one because of the attention check))

# When computing IICMs and processing SSMs (in the data_analysis.R script), we will also remove item 1 since it can't be compared across scale versions.

combined <- combined %>%
    select(-c("FOMO (OG)_11...38", "FOMO (SP)_6...108", "FOMO (LP)_6...130")) %>%
    select(-c("FOMO (OG)_1")) %>%
    rename("FOMO (OG)_11" = "FOMO (OG)_11...41",
           "FOMO (SP)_6" = "FOMO (SP)_6...113",
           "FOMO (LP)_6" = "FOMO (LP)_6...135") %>%
    rename("FOMO (OG)_8" = "FOMO (OG)_9",
            "FOMO (OG)_9" = "FOMO (OG)_10",
            "FOMO (OG)_10" = "FOMO (OG)_11")
### End of FOMO cleaning intermission ###


# Check if any participants have identical responses to all items in a scale (e.g., they answered each item with a '4')
scale_names <- unique(gsub("_.*", "", names(combined[6:146])))
identical_responders <- c()
identical_responses <- c()
for (row in 1:nrow(combined)){
  for (scale in scale_names){
    check_row <- combined[row,] %>%
      select(starts_with(scale))
    if (sum(is.na(check_row)) == length(check_row)){
      next # skip if the participant didn't answer any items in the scale
    } else {
      if (var(t(check_row), na.rm = TRUE) == 0){ # check if the participant answered all items in the scale with the same value
        identical_responders <- c(identical_responders, row)
        identical_responses <- c(identical_responses, scale)
      } 
    }
  }
}

# Remove rows that have identical responses to all items in a scale
combined <- combined %>%
  filter(!row_number() %in% unique(identical_responders))

## Reverse score contraindicative items
# FOMO - 2, 3 (8-point scale)
# FNE - 2, 4, 7, 9 (5-point scale)
# C - 6, 7, 8, 9, 10 (5-point scale)
# PE - 7, 8, 9 (5-point scale)
combined <- combined %>%
  mutate(across(c(starts_with("FOMO") & ends_with(c("2", "3"))), ~ 9 - as.numeric(.))) %>%
  mutate(across(c(starts_with("FNE") & ends_with(c("2", "4", "7", "9"))), ~ 6 - as.numeric(.))) %>%
  mutate(across(c(starts_with("C") & ends_with(c("6", "7", "8", "9", "10"))), ~ 6 - as.numeric(.))) %>%
  mutate(across(c(starts_with("PE") & ends_with(c("7", "8", "9"))), ~ 6 - as.numeric(.)))


# Write the cleaned data to a csv file
write.csv(combined, "jul_13_full_cleaned.csv", row.names = FALSE)
