# Load required libraries
library(tidyverse)
library(lubridate)
library(patchwork)


# Load the data
prices <- read_csv("all_historical_prices2.csv")
sentiment <- read_csv("all_news_sentiments_v3.csv")

sentiment_line <- sentiment %>%
  filter(symbol == "AAPL") %>%
  select(symbol, sentiment, sentimentScore, publishedDate) %>% 
  mutate(publishedDate = as_datetime(publishedDate)) %>%
  mutate(date = date(publishedDate)) %>% 
  group_by(date) %>%
  summarize(avg_sentiment = mean(sentimentScore))

prices <- prices %>%
  # mutate(date = mdy(date)) %>%
  select(date, close)

combined_data <- prices %>%
  left_join(sentiment_line, by = "date")

# Calculate daily average sentiment for the line plot
daily_avg_sentiment <- sentiment %>%
  group_by(date) %>%
  summarize(avg_sentiment = mean(sentimentScore))

# Create the plot
linechart <- ggplot(combined_data, aes(x = date)) +
  geom_line(aes(y = close, color = "Stock Price"), size = 1) +
  geom_line(aes(y = avg_sentiment * 100 + 200, color = "Sentiment"), size = 1) +
  scale_y_continuous(
    name = "Stock Price ($)",
    sec.axis = sec_axis(~(. - 200) / 100, name = "Average Sentiment Score")
  ) +
  scale_color_manual(values = c("Stock Price" = "blue", "Sentiment" = "red")) +
  labs(
    title = "AAPL Stock Price vs Sentiment Analysis",
    x = "Date",
    color = "Legend"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 10))

# Process sentiment data
sentiment_dotplot <- sentiment %>% 
  filter(symbol == 'AAPL') %>% 
  mutate(datetime = as_datetime(publishedDate),
         date = as_date(publishedDate)) %>%
  select(datetime, date, sentiment, sentimentScore)

# Create the plot
dotplot <- ggplot(sentiment_dotplot, aes(x = date, fill = factor(sentiment) )) +
  geom_dotplot(stackgroups = TRUE, 
               binwidth = .1, 
               binpositions = 'all'
               # stackdir = 'center',
               # position = position_dodge(width = 0.7)
               ) +
  # geom_line(data = combined_data, aes(x = date, y = close / max(close) * 100), color = "black", size = 1) +
  # scale_y_continuous(sec.axis = sec_axis(~ . * max(combined_data$close) / 100, name = "Stock Price ($)")) +
  scale_fill_manual(values = c("Positive" = "green","Neutral" = "gray", "Negative" = "red")) +
  theme_minimal() +
  labs(title = "Sentiment Analysis of AAPL News Articles",
       x = "Date",
       y = "Sentiment") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_x_date(date_breaks = "1 day", date_labels = "%d %b %Y") +
  theme(legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 10))


# Combine the two plots using patchwork
combined_plot <- linechart / dotplot

# Display the combined plot
print(combined_plot)



