import pandas as pd
prediction_data = pd.read_csv("predictions_adversarial_vgg_best.csv")
white = prediction_data[prediction_data.race == 0]
nonwhite = prediction_data[prediction_data.race != 0]

female = prediction_data[prediction_data.label == 0]
male = prediction_data[prediction_data.label == 1]

white_female = white[white.label == 0]
white_male = white[white.label == 1]
nonwhite_female = nonwhite[nonwhite.label == 0]
nonwhite_male = nonwhite[nonwhite.label == 1]

print("Total Accuracy:")
print(sum(prediction_data.label == prediction_data.pred)/len(prediction_data))

#gender split
def confusion(male, female, label1, label2):
    #condition: gender: present (1 is male), absent is female (0)
    tp_men = sum(male.label == male.pred)
    fn_men = sum(male.label != male.pred)
    fp_men = sum(female.label != female.pred)
    tn_men = sum(female.label == female.pred)
    tpr_men = tp_men/(tp_men + fn_men) #sum(male.label == male.pred)/(sum(male.label == male.pred) + sum(female.label != female.pred))
    #false positive rate is 1 - specificity
    fpr_men = 1 - tn_men/(tn_men+fp_men)#sum(male.label != male.pred)/(sum(female.label == female.pred) + sum(male.label != male.pred))
    ppv_men = tp_men/(tp_men + fp_men)

    tp_women = sum(female.label == female.pred)
    fn_women = sum(female.label != female.pred)
    fp_women = sum(male.label != male.pred)
    tn_women = sum(male.label == male.pred)
    tpr_women = tp_women/(tp_women + fn_women)
    fpr_women = 1 - tn_women/(tn_women+fp_women)#sum(male.label != male.pred)/(sum(female.label == female.pred) + sum(male.label != male.pred))
    ppv_women = tp_women/(tp_women + fp_women)

    print(label1)
    print("PPV", ppv_men)
    print("ERR", 1 - ppv_men)
    print("TPR:", tpr_men)
    print("FPR:", fpr_men)

    print(label2)
    print("PPV", ppv_women)
    print("ERR", 1 - ppv_women)
    print("TPR:", tpr_women)
    print("FPR:", fpr_women)

# def confusion_category(data):
#     #0 is female
#     #1 is male
#     correct = sum(data.label == data.pred)
#     acc = correct/len(data)
#
#     # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
#     TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
#
#     # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
#     FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
#
# # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
# FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
#     print("ACCURACY:", acc)

confusion(male, female, "male", "female")

confusion(nonwhite_male, nonwhite_female, "nonwhite_male", "nonwhite_female")

confusion(white_male, white_female, "white_male", "white_female")
