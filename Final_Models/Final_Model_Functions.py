# Emotion Prediction model based on words.
def word_class_model(df, df_sentences, word_df_name):
    pos_emo_list = [] # positive emotion list, I'm going to put the non-negative weights in here
    neg_emo_list = [] # negative emotion list, I'm going to put the negative weights in here
    temp_df = df.copy()
    # function to determine weight per sentence/row.
    def find_weights(sentence):
        pos_emo = 0
        neg_emo = 0
        for word in sentence.split():
            if (word + " ") in word_df_name.index: # there was a trailing white space for the words in the index
                neg_emo += word_df_name.loc[(word + ' ')]['disgust'] # Grab the weight
                neg_emo += word_df_name.loc[(word + ' ')]['anger']
                neg_emo += word_df_name.loc[(word + ' ')]['sad']
                neg_emo += word_df_name.loc[(word + ' ')]['fear']
                pos_emo += word_df_name.loc[(word + ' ')]['happy']
                pos_emo += word_df_name.loc[(word + ' ')]['surprise']
                pos_emo += word_df_name.loc[(word + ' ')]['neutral']
        pos_emo_list.append(pos_emo)
        neg_emo_list.append(neg_emo)
    temp_df[df_sentences].apply(find_weights) # applying the function should create two lists the same length as our data
    # Incorporate lists into dataframe
    temp_df['Positive_Word_Weight'] = pos_emo_list
    temp_df['Negative_Word_Weight'] = neg_emo_list
    # Compare the columns
    Word_Predictions_Mask = temp_df['Positive_Word_Weight'] < temp_df['Negative_Word_Weight']
    # create a list for our final predictionc
    Final_Predictions = []
    # Convert Trues and Falses into 1s and -1s.
    for truthy in Word_Predictions_Mask:
        if truthy == True:
            Final_Predictions.append(1) # If negative
        else:
            Final_Predictions.append(-1) # If anything but negative
    # return final prediction
    return Final_Predictions

# Final model predictor
# final model function
def combined_model_predictor(df_to_predict, feature, word_df_name, final_model_list):
        def word_class_model(df, df_sentences, word_df_name):
            pos_emo_list = [] # positive emotion list, I'm going to put the non-negative weights in here
            neg_emo_list = [] # negative emotion list, I'm going to put the negative weights in here
            temp_df = df.copy()
            # function to determine weight per sentence/row.
            def find_weights(sentence):
                pos_emo = 0
                neg_emo = 0
                for word in sentence.split():
                    if (word + " ") in word_df_name.index: # there was a trailing white space for the words in the index
                        neg_emo += word_df_name.loc[(word + ' ')]['disgust'] # Grab the weight
                        neg_emo += word_df_name.loc[(word + ' ')]['anger']
                        neg_emo += word_df_name.loc[(word + ' ')]['sad']
                        neg_emo += word_df_name.loc[(word + ' ')]['fear']
                        pos_emo += word_df_name.loc[(word + ' ')]['happy']
                        pos_emo += word_df_name.loc[(word + ' ')]['surprise']
                        pos_emo += word_df_name.loc[(word + ' ')]['neutral']
                pos_emo_list.append(pos_emo)
                neg_emo_list.append(neg_emo)
            temp_df[df_sentences].apply(find_weights) # applying the function should create two lists the same length as our data
            # Incorporate lists into dataframe
            temp_df['Positive_Word_Weight'] = pos_emo_list
            temp_df['Negative_Word_Weight'] = neg_emo_list
            # Compare the columns
            Word_Predictions_Mask = temp_df['Positive_Word_Weight'] < temp_df['Negative_Word_Weight']
            # create a list for our final predictionc
            Final_Predictions = []
            # Convert Trues and Falses into 1s and -1s.
            for truthy in Word_Predictions_Mask:
                if truthy == True:
                    Final_Predictions.append(1) # If negative
                else:
                    Final_Predictions.append(-1) # If anything but negative
            # return final prediction
            return Final_Predictions
        predictions = []
        for model in final_model_list:
            predictions.append(model.predict(df_to_predict[feature]))
        word_model = word_class_model(df_to_predict, feature, word_df_name)
        predictions.append(word_model)
        predicted_weights = ((np.array(predictions[0])*emot_long_weight)+
                                (np.array(predictions[1])*pos_neg_weight) +
                                (np.array(predictions[2])*emot_short_weight) +
                              (np.array(predictions[3])*word_weights))
        final_predictions = []
        for pred in predicted_weights:
                if pred > 0:
                    final_predictions.append(1)
                else:
                    final_predictions.append(-1)
        return final_predictions
