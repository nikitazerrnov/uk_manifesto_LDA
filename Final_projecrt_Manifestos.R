#packages installing and loading

packages <- c("textmineR", "viridis", "stm", "topicmodels",
                      "quanteda", "readtext", "topicdoc", "readr",
                      "base", "tidyverse", "tidytext", "topicmodels",
                      "purrr", "tm", "textstem", "stringr", "data.table", "reshape2")

installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

invisible(lapply(packages, library, character.only = TRUE))

#loading the data


con1992 <- read_csv("uk_manifestos/con1992.csv") %>% select(text)
lab1992 <- read_csv("uk_manifestos/labr1992.csv") %>% select(text)

con1997 <- read_csv("uk_manifestos/con1997.csv") %>% select(text)
lab1997 <- read_csv("uk_manifestos/lab1997.csv") %>% select(text)

con2001 <- read_csv("uk_manifestos/con2001.csv") %>% select(text)
lab2001 <- read_csv("uk_manifestos/lab_2001.csv") %>% select(text) 
lab2001 <- as.data.frame(paste(lab2001$text, collapse = " "))
names(lab2001)[1] <- "text"


con2005 <- read_csv("uk_manifestos/con2005.csv") %>% select(text)
lab2005 <- read_csv("uk_manifestos/lab2005.csv") %>% select(text)

lab2010c <- readChar("uk_manifestos/lab2010c.txt", file.info("uk_manifestos/lab2010c.txt")$size) %>% 
  as.data.frame()
names(lab2010c)[1] <- "text"

con2010c <- readChar("uk_manifestos/con2010c.txt", file.info("uk_manifestos/con2010c.txt")$size) %>% 
  as.data.frame()
names(con2010c)[1] <- "text"

con2017 <- read_csv("uk_manifestos/con2017.csv") %>% select(text)
lab2017 <- read_csv("uk_manifestos/lab2017.csv") %>% select(text)
lab2017 <- as.data.frame(paste(lab2017$text, collapse = " "))
names(lab2017)[1] <- "text"

con2017 <- as.data.frame(paste(con2017$text, collapse = " "))
names(con2017)[1] <- "text"

con2019 <- read_csv("uk_manifestos/con2019.csv") %>% select(text)
lab2019 <- read_csv("uk_manifestos/lab2019.csv")%>% select(text)
lab2019 <- as.data.frame(paste(lab2019$text, collapse = " "))
names(lab2019)[1] <- "text"

con2019 <- as.data.frame(paste(con2019$text, collapse = " "))
names(con2019)[1] <- "text"


#vertically concatenate all manifestos
manifestos_all = rbind(con1992, con1997, con2001, con2005, con2010c, con2017, con2019,
                       lab1992, lab1997, lab2001, lab2005, lab2010c, lab2017, lab2019)
manifestos_all$name <- c(
  "con1992", "con1997", "con2001", "con2005", "con2010", "con2017", "con2019",
  "lab1992", "lab1997", "lab2001", "lab2005", "lab2010", "lab2017", "lab2019")

#creating metadata - extracting year and party
manifestos_all$year = str_extract(manifestos_all$name, "[0-9]+") %>% as.factor()
manifestos_all <- manifestos_all %>%
  mutate(party = ifelse(str_detect(name, "con"), "Conservatives", "Labour"))
manifestos_all$party = as.factor(manifestos_all$party)
#settitng up id column
manifestos_all <- manifestos_all %>% mutate(doc_id = row_number())



manifestos_simple = manifestos_all %>% select(text, doc_id)

#Tokenisation, cleansing
text_cleaning_tokens <- manifestos_simple %>% 
  tidytext::unnest_tokens(word, text)
text_cleaning_tokens$word = lemmatize_words(text_cleaning_tokens$word)
text_cleaning_tokens$word <- str_remove_all(text_cleaning_tokens$word, "[[:digit:]]")
text_cleaning_tokens$word <- str_remove_all(text_cleaning_tokens$word, "[[:punct:]]")
text_cleaning_tokens <- filter(text_cleaning_tokens, !(nchar(word) == 1) & !(word=="")) %>% 
  anti_join(stop_words) 

text_cleaning_tokens_count = text_cleaning_tokens %>% 
  group_by(word) %>% summarise(cnt = n())


#figure 1 in report
text_cleaning_tokens_count %>% slice_max(cnt, n = 10) %>% 
  ggplot() + geom_col(aes(x = reorder(word, cnt), y = cnt, fill = cnt)) + coord_flip() +
  theme_bw() + scale_fill_gradient(low = "blue", high = "red") + labs(
    title = 'Top 10 words used in corpus',
    fill = 'No. of \nmentions',
    y = 'count',
    x = 'words',
    caption = "Source: Manifesto Database, \nauthor's calculations"
  )


#only use words between 5th and 95 percentile in frequency
text_cleaning_tokens_count = filter(text_cleaning_tokens_count,
  text_cleaning_tokens_count$cnt > quantile(text_cleaning_tokens_count$cnt, 0.05) &
  text_cleaning_tokens_count$cnt < quantile(text_cleaning_tokens_count$cnt, 0.95))
text_cleaning_tokens = filter(text_cleaning_tokens, word %in% text_cleaning_tokens_count$word)

tokens <- text_cleaning_tokens

##########spead matrix 

tokens <- tokens %>% group_by(doc_id) %>% mutate(ind = row_number()) %>%
  tidyr::spread(key = ind, value = word)
tokens [is.na(tokens)] <- ""
tokens <- tidyr::unite(tokens, text,-doc_id,sep =" " ) 
tokens$text <- trimws(tokens$text)

#create Docuemnt term matrix
dtm <- CreateDtm(tokens$text, 
                 doc_names = tokens$doc_id)
# frequency
tf <- TermDocFreq(dtm = dtm)
original_tf <- tf %>% select(term, term_freq,doc_freq)
original_tf = as.data.frame(original_tf)
rownames(original_tf) <- 1:nrow(original_tf)
#vocab setting (needed as argument to fit lda model)
vocabulary <- tf$term
dtm = dtm

#this segment is inspired by (the code copied from)
#https://towardsdatascience.com/beginners-guide-to-lda-topic-modelling-with-r-e57a5a8e7a25

k_list <- seq(1, 22, by = 1)
model_dir <- paste0("models_", digest::digest(vocabulary, algo = "sha1"))
if (!dir.exists(model_dir)) dir.create(model_dir)
model_list <- TmParallelApply(X = k_list, FUN = function(k){
  filename = file.path(model_dir, paste0(k, "_topics.rda"))
  
  if (!file.exists(filename)) {
    m <- FitLdaModel(dtm = dtm, k = k, iterations = 500)
    m$k <- k
    m$coherence <- CalcProbCoherence(phi = m$phi, dtm = dtm, M = 5)
    save(m, file = filename)
  } else {
    load(filename)
  }
  
  m
}, export=c("dtm", "model_dir")) # export for Windows machines

#model tuning
#choosing the best model
coherence_mat <- data.frame(k = sapply(model_list, function(x) nrow(x$phi)), 
                            coherence = sapply(model_list, function(x) mean(x$coherence)), 
                            stringsAsFactors = FALSE)
#plotting the coherence plot (appendix 2 in the report) for choosing the best K 

ggplot(coherence_mat, aes(x = k, y = coherence)) +
  geom_point() +
  geom_line(group = 1)+
  ggtitle("Best # Topics by Coherence Score") + theme_minimal() +
  scale_x_continuous(breaks = seq(1,20,1)) + ylab("Coherence")

model <- model_list[which.max(coherence_mat$coherence)][[ 1 ]]

####having determined the appropriate value of K i re-built the model using another packaged
#the difference is in the class of the object produced LDA() is easier to work with

model <- topicmodels::LDA(dtm, k = 12,
                          method= "Gibbs",
                          control = list(seed = 1234))
man_topics <- tidy(model, matrix = "beta")

#most probable words for each topic (appendix 2 in report)
man_topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  arrange(topic, -beta) %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() + theme_minimal()
  
beta_prob = man_topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 5) %>% 
  ungroup() %>%
  arrange(topic, -beta)
beta_prob %>% group_by(topic) %>% 


# examine the per-document-per-topic probabilities 

man_documents <- tidy(model, matrix = "gamma")

man_documents$document = as.numeric(man_documents$document)
manifestos_all = select(manifestos_all, - text)
gam_mod = left_join(man_documents, manifestos_all, by = c("document" = "doc_id"))

#assigning proper topic names
gam_mod$topic_name = ifelse(gam_mod$topic == 1, 'Housing',
                            ifelse(gam_mod$topic == 2, 'Level up',
                                   ifelse(gam_mod$topic == 3, 'Science',
                                          ifelse(gam_mod$topic == 4, 'Commerce',
                                                 ifelse(gam_mod$topic == 5, 'Cost-cutting',
                                                        ifelse(gam_mod$topic == 6, 'Democracy',
                                                               ifelse(gam_mod$topic == 7, 'Education',
                                                                      ifelse(gam_mod$topic == 8, 'Governing',
                                                                             ifelse(gam_mod$topic == 9, 'NHS',
                                                                                    ifelse(gam_mod$topic == 10, 'Independence & Devolution',
                                                                                           ifelse(gam_mod$topic == 11, 'Digitalisation','Equality')))))))))))


#independence BREXIT
gam_mod %>% filter(topic == 10) %>% ggplot() + geom_line(aes(x = year, y = gamma, group = 1)) + 
  facet_grid(rows = vars(party)) +
  theme_minimal() + labs(title = 'The Share of Independence Topic',
                         subtitle = 'Inside the Manifestos of Both Parties',
                         caption = "Source: Manifesto Database, \nauthor's calculations"
                         )

#setting up colour palette - catergorical 12 colours
palet = c('#8a3ffc', '#33b1ff', '#007d79', '#ff7eb6', '#fa4d56', '#fff1f1', '#6fdc8c', '#4589ff', '#d12771', '#d2a106',
          '#08bdba', '#bae6ff')

#main findings plot
am_mod  %>% ggplot() + geom_col(aes(x = as.factor(name), y = gamma, fill = topic_name)) + coord_flip() + theme_minimal() +
  scale_fill_manual(values = palet) + labs(
    title = 'The Share of Topics in All Manifestos',
    caption = "Source: Manifesto Database, \nauthor's calculations",
    x = 'Manifesto',
    y = 'gamma',
    fill = ""
  )

#clear environment and finish the session
rm(list = ls())
dev.off()



