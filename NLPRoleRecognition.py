# =================================================================
# PROJECT: Fine-tune PhoBert for Role Recognition
# PURPOSE: Fine-tune PhoBert model for role recognition tasks
# AUTHOR: K35 - Khoa Học Tích Hợp - Nhóm 2
# DATE: 2026
# =================================================================
import os
import Tfidf
import FinetunePhoBert
import TrainByContext
import argparse
import NormalizeData

# =================================================================
# Define the NLPRoleRecognition class to orchestrate the NLP role recognition pipeline
# =================================================================
class NLPRoleRecognition:

    # =================================================================
    # Initialize the NLPRoleRecognition class with instances of Tfidf, FinetunePhoBert, and TrainByContext
    # =================================================================
    def __init__(self):
        self.tfidf_model = Tfidf.Tfidf()
        self.finetune_phobert = FinetunePhoBert.FinetunePhoBert()
        self.train_by_context = TrainByContext.TrainByContext()
        self.normalizeData = NormalizeData.NormalizeData()

    # =================================================================
    # Parse command-line arguments for the NLP role recognition pipeline
    # =================================================================
    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="NLP Role Recognition")
        parser.add_argument("--option", type=str, required=True, help="Option such as: Train, Normalize or Predict")
        parser.add_argument("--optionType", type=str, required=True, help="It depends on the option. For 'Train', it could be 'TF-IDF', 'PhoBert', or 'BiLSTM'. For 'Normalize', it should be the folder containing all of the Parquet files. For 'Predict', it should be the file containing the utterance to predict.")
        return parser.parse_args()

    # =================================================================
    # Main function to run the NLP Role Recognition pipeline
    # =================================================================
    def main(self):
        args = self.parse_arguments()
        args.option = args.option.capitalize()
        args.optionType = args.optionType.capitalize()

        if args.option not in ["Train", "Predict", "Normalize"]:
            print("Invalid option. Please specify either 'Train' or 'Predict' or 'Normalize'.")
            return

        match args.option:
            case "Train":
                if args.optionType not in ["TF-IDF", "PhoBert", "BiLSTM"]:
                    print("Invalid option type for training. Please specify 'TF-IDF', 'PhoBert', or 'BiLSTM'.")
                if args.optionType == "TF-IDF":
                    self.tfidf_model.train()
                elif args.optionType == "PhoBert":
                    self.finetune_phobert.train()
                elif args.optionType == "BiLSTM":
                    self.train_by_context.train()
                return
            case "Normalize":
                # checking exisiting file
                if not os.path.exists(args.optionType):
                    print("The specified file for normalization does not exist.")

                # normalize data with the input folder
                self.normalizeData.scanFileAndNormalize(args.optionType)
                return
            case _: #Predict
                # checking exisiting file
                if not os.path.exists(args.optionType):
                    print("The specified file for prediction does not exist.")

                # process predict
                if args.optionType.endswith(".csv"):
                    self.tfidf_model.predict(args.optionType)
                elif args.optionType.endswith(".txt"):
                    self.finetune_phobert.predict(args.optionType)
                else:
                    print("Invalid file format for prediction. Please provide a .csv or .txt file.")
                return

# =================================================================
# Run the NLPRoleRecognition pipeline if this script is executed directly
# =================================================================
if __name__ == "__main__":
    nlp_role_recognition = NLPRoleRecognition()
    nlp_role_recognition.main()