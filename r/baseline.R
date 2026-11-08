# Baseline predictions in R, for students who prefer R to Python.
#
# Does the same as `python run.py predict` with the two starting models:
#   list-aware: every home sells for exactly its list price
#   off-market: the median sold price of the 10 nearest earlier sales of the same type
# and writes the same two files: submissions/submission_off_market.csv and
# submissions/submission_list_aware.csv.
#
# 1. Download listings.csv, test_off_market.csv and test_list_aware.csv from
#    https://huggingface.co/datasets/Baliyang/SSU_data_dash ("Files and versions" tab).
# 2. Set data_dir below to the folder they are in.
# 3. From the repo folder, run:  Rscript r/baseline.R   (or open it in RStudio and click Source)
# Only base R is needed. data.table::fread() reads the files faster, if you have it.

data_dir <- "data"          # folder with the three CSV files
out_dir <- "submissions"    # where the two submission files go
k <- 10                     # how many nearby sales to take the median of
options(scipen = 999)       # write prices as 1250000, not 1.25e+06

read_data <- function(name) {
  df <- read.csv(file.path(data_dir, paste0(name, ".csv")), stringsAsFactors = FALSE, encoding = "UTF-8")
  for (column in intersect(c("date_listed", "date_sold"), names(df))) {
    df[[column]] <- as.Date(substr(df[[column]], 1, 10))
  }
  df
}

listings <- read_data("listings")
off_market <- read_data("test_off_market")
list_aware <- read_data("test_list_aware")

# Newest sales first, so when several sales are equally near (a condo building
# shares one location), the most recent ones are picked.
listings <- listings[order(-as.numeric(listings$date_sold), listings$ml_num, method = "radix"), ]

# Off-market baseline for one home: only sales from before the home was listed,
# so the model never looks into the future.
nearest_sales_price <- function(home) {
  earlier <- listings$date_sold < home$date_listed
  pool <- earlier & listings$house_type_name == home$house_type_name
  if (!any(pool)) pool <- earlier   # no earlier sale of this type: use every type
  # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
  # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
  dx <- (listings$longitude[pool] - home$longitude) * 0.72
  dy <- listings$latitude[pool] - home$latitude
  distance <- sqrt(dx^2 + dy^2)
  nearest <- head(order(distance), k)   # ties keep the table order: newest first
  median(listings$sold_price[pool][nearest])
}

write_submission <- function(homes, predicted, track) {
  dir.create(out_dir, showWarnings = FALSE)
  path <- file.path(out_dir, paste0("submission_", track, ".csv"))
  out <- data.frame(ml_num = homes$ml_num, predicted_price = round(predicted))
  write.csv(out, path, row.names = FALSE, quote = FALSE)
  cat("Wrote", path, "with", nrow(homes), "homes\n")
}

off_market_predictions <- sapply(seq_len(nrow(off_market)), function(i) nearest_sales_price(off_market[i, ]))
write_submission(off_market, off_market_predictions, "off_market")
write_submission(list_aware, list_aware$list_price, "list_aware")
