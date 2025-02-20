
# basic imports
import os 
import numpy as np
import wandb

# utils imports
import scipy.special as special
import sklearn.metrics as metrics

# display imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




def roc_curve(wandb_run, softmax_preds, targets, target_names, folder_path, log_scale=False, plot_name='roc_curve', figsize=(10, 6)):
    """
    folder_path (str) : Folder where to save the figure
    plot_name   (str) : Name to give to the plot when saving at folder_path & logging in wandb
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Display performances of a random classifier
    ax.plot([1e-4, 10], [1e-4, 10], 'k--', label='Random Guess', lw=1)


    # --- Computing the ROC curves --- #
    for i, target_name in enumerate(target_names):
        signal_class_preds = softmax_preds[:, i]

        # Compute fpr, tpr and corresponding auc
        fpr, tpr, _ = metrics.roc_curve(targets == i, signal_class_preds)
        roc_auc = metrics.auc(fpr, tpr)

        # Plots
        if log_scale:
            # Handling of 0 values if any
            fpr = np.where(fpr <= 5e-4, 5e-4, fpr)
            tpr = np.where(tpr <= 5e-4, 5e-4, tpr)

        ax.plot(fpr, tpr, label=f"Signal: {target_name}, AUC :{roc_auc:.2f}", lw=0.8)
        ax.legend()


    # Increase appearence 
    plt.title(f"ROC curves")
    ax.set_xlim([1e-4, 10]) 
    ax.set_ylim([1e-4, 10])

    if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')

    plt.grid(True, which='both',  ls='--', lw=0.5)

    plt.xlabel('Background acceptance')
    plt.ylabel('Signal efficiency')
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{folder_path}/{plot_name}.png", dpi=300)
    plt.close(fig)

    # Reopen the image and log to wandb
    if wandb_run is not None:
        image = plt.imread(f"{folder_path}/{plot_name}.png")
        wandb_run.log({plot_name: wandb.Image(image)})


def zoomed_roc_curve(wandb_run, softmax_preds, targets, target_names, folder_path, log_scale=False, plot_name='roc_curve', figsize=(10, 6)):

        fig, ax = plt.subplots(figsize=figsize)

        # --- Computing the ROC curves --- #
        for i, target_name in enumerate(target_names):
            signal_index = i

            signal_class_preds = softmax_preds[:, signal_index]

            # Compute fpr, tpr and corresponding auc
            fpr, tpr, _ = metrics.roc_curve(targets == signal_index, signal_class_preds)
            roc_auc = metrics.auc(fpr, tpr)

            # Plots
            if log_scale:
                # Handling of 0 values if any
                fpr = np.where(fpr <= 5e-4, 5e-4, fpr)
                tpr = np.where(tpr <= 5e-4, 5e-4, tpr)

            ax.plot(fpr, tpr, label=f"Signal: {target_name}, AUC :{roc_auc:.2f}", lw=0.8)
            ax.legend()


        # Increase appearence 
        plt.title(f"Zoomed ROC curves")
        ax.set_xlim([7e-4, 2e-1]) 
        ax.set_ylim([0.8, 1.2])

        if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')

        plt.grid(True, which='both',  ls='--', lw=0.5)
        plt.hlines(0.995,0, 1, colors='k', linestyles='--', lw=0.5, label="99.5% signal efficiency")
        plt.hlines(0.99,0, 1, colors='k', linestyles='--', lw=0.5, label="99% signal efficiency")
        plt.vlines(0.05, 0, 1, colors='k', linestyles='-', lw=0.5, label="5% acceptance")
        plt.vlines(0.005, 0, 1, colors='k', linestyles='-', lw=0.5, label="0.5% acceptance")
        plt.vlines(0.01, 0, 1, colors='k', linestyles='-', lw=0.5, label="1% acceptance")

        plt.xlabel('Background acceptance')
        plt.ylabel('Signal efficiency')

        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f"{folder_path}/{plot_name}.png", dpi=300)
        plt.close(fig)

        # Reopen the image and log to wandb
        if wandb_run is not None:
            image = plt.imread(f"{folder_path}/{plot_name}.png")
            wandb_run.log({plot_name: wandb.Image(image)})


def p_r_curve(wandb_run, data, targets, signal_key, folder_path, log_scale=False, plot_name='p_r_curve', figsize=(10, 6)):
    """
    preds   (array) (n_samples, ) : 1-d array of the predictions
    targets (array) (n_samples, ) : 1-d array of the targets

    folder_path (str) : Folder where to save the figure
    plot_name   (str) : Name to give to the plot when saving at folder_path & logging in wandb

    """
    # Softmax the raw_predicitons + take only the "signal" inputs
    preds = data[signal_key]

    precision, recall, _ = metrics.precision_recall_curve(targets, preds)
    average_precision_score = metrics.average_precision_score(targets, preds)

    # Plot
    fig, ax = plt.subplots(figsize=figsize) 
    if log_scale:
        precision = np.where(precision <= 1e-8, 1e-8, precision)
        recall    = np.where(recall <= 1e-8, 1e-8, recall) 

        ax.set_xscale('log')
        ax.set_yscale('log')

    # Plot
    ax.plot(recall, precision, linewidth=2, label=f'AP : {average_precision_score}')

    # Increase appearence of the plot 
    plt.title(f"Precision - Recall for {signal_key[-2:]} as signal")
    
    ax.set_xlim([1e-10, 100])  # Extend limits
    ax.set_ylim([1e-10, 100])
    ax.grid(True, which="both")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    # Save the figure
    plt.savefig(f"{folder_path}/{plot_name}.png", dpi=300)
    plt.close(fig)

    # Reopen the image and log to wandb
    if wandb_run is not None:
        image = plt.imread(f"{folder_path}/{plot_name}.png")
        wandb_run.log({plot_name: wandb.Image(image)})




def confusion_matrix(wandb_run, predicted_classes, targets, target_names, folder_path, plot_name, figsize=(10, 6)):

    cm = metrics.confusion_matrix(targets, predicted_classes)

    df_cm = pd.DataFrame(
        cm,
        index=target_names,
        columns=target_names
    )

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_cm, ax=ax, annot=True, linewidths=0.8)

    # Save the figure
    plt.savefig(f"{folder_path}/{plot_name}.png", dpi=300)
    plt.close(fig)

    # Reopen the image and log to wandb
    if wandb_run is not None:
        image = plt.imread(f"{folder_path}/{plot_name}.png")
        wandb_run.log({plot_name: wandb.Image(image)})

def scatplot_2d(wandb_run, data, x, y, folder_path, plot_name='2d_scatplot', figsize=(10, 6)):
    """
    preds   (n_samples, 2) # For binary classification. Can be extended to more if wanted but you'll have to change the code
    targets (n_samples, )
    """

    fig, ax = plt.subplots(figsize=figsize) 
    ax = sns.scatterplot(data=data, x=x, y=y, hue='target_names', s=10, ax=ax)

    # Increase apperance
    plt.title('Model output with true labels')
    #plt.xlabel('1st dimension')
    #plt.ylabel('2nd dimension')
    plt.grid(True)
    plt.legend()

    # Save the figure
    plt.savefig(f"{folder_path}/{plot_name}.png", dpi=300)
    plt.close(fig)

    # Reopen the image and log to wandb
    if wandb_run:
        image = plt.imread(f"{folder_path}/{plot_name}.png")
        wandb_run.log({plot_name: wandb.Image(image)})

def histogram_2d(
        wandb_run, 
        data, 
        x, 
        y,
        folder_path, 
        plot_name, 
        log_scale=(False, False), 
        figsize=(10, 6)
    ):

    fig, ax = plt.subplots(figsize=figsize)

    # Plot
    ax = sns.histplot(data, x=x, y=y, hue="target_names", log_scale=log_scale, cbar=True, ax=ax)

    plt.title("Model 2d output with true labels")
    plt.grid(True)
    plt.legend()

    # Save the figure
    plt.savefig(f"{folder_path}/{plot_name}.png", dpi=300)
    plt.close(fig)

    # Reopen the image and log to wandb
    if wandb_run:
        image = plt.imread(f"{folder_path}/{plot_name}.png")
        wandb_run.log({plot_name: wandb.Image(image)})

def combined_histograms_plot(
        wandb_run, 
        data, 
        folder_path, 
        plot_name, 
        log_yscale=False, 
        fill=True,
        element='bars',
        figsize=(10, 6)):

    fig, ax = plt.subplots(figsize=figsize)

    # Plot
    for col in data.columns:
        label = str(col)  
        ax = sns.histplot(data[col], label=label, fill=fill, element=element, ax=ax)

    # Increase apperance
    if log_yscale:
        ax.set_yscale('log')

    plt.title("Empirical distributions of over model's outputs")
    plt.grid(True)
    plt.legend()

    # Save the figure
    plt.savefig(f"{folder_path}/{plot_name}.png", dpi=300)
    plt.close(fig)

    # Reopen the image and log to wandb
    if wandb_run:
        image = plt.imread(f"{folder_path}/{plot_name}.png")
        wandb_run.log({plot_name: wandb.Image(image)})


def preds_targets_histogram(
        wandb_run, 
        preds, 
        targets,
        folder_path, 
        plot_name, 
        target_name,
        log_yscale=False, 
        fill=True,
        element='bars', 
        figsize=(10, 6)
    ):
    fig, ax = plt.subplots(figsize=figsize)

    # Plot   
    ax = sns.histplot(preds, label='model out', fill=fill, element=element, ax=ax)
    ax = sns.histplot(targets, label='targets', fill=fill, element=element, ax=ax)

    # Increase apperance
    if log_yscale:
        ax.set_yscale('log')

    plt.title(f"Model outputs vs targets for {target_name}")
    plt.grid(True)
    plt.legend()

    # Save the figure
    plt.savefig(f"{folder_path}/{plot_name}.png", dpi=300)
    plt.close(fig)

    # Reopen the image and log to wandb
    if wandb_run:
        image = plt.imread(f"{folder_path}/{plot_name}.png")
        wandb_run.log({plot_name: wandb.Image(image)})



def count_plot(
    wandb_run,
    preds,
    target_names,
    folder_path,
    plot_name,
    figsize=(10, 6)
):
    fig, ax = plt.subplots(figsize=figsize)

    prediction_with_names = [target_names[i] for i in np.argmax(preds, axis=1)]
    # Plot
    sns.countplot(prediction_with_names, ax=ax)

    # Save the figure
    plt.savefig(f"{folder_path}/{plot_name}.png", dpi=300)
    plt.close(fig)

    # Reopen the image and log to wandb
    if wandb_run:
        image = plt.imread(f"{folder_path}/{plot_name}.png")
        wandb_run.log({plot_name: wandb.Image(image)})
    
    


""" Archive """

def wandb_roc_curve(wandb_run, targets, preds, softmax=False, labels=None, plot_name="roc_curve"):
    """
    The plot_name represents the id of the roc curve on wandb. Do NOT change it
    over the runs.
    """
    # Love ListConfig (cannot call label[np.int64] when label is a ListConfig)
    # convert & reshape to right type for wandb

    print(preds, targets)
    if not softmax:
        preds = special.softmax(preds, axis=1)

    print(preds.shape)
    print(preds)
    print(targets)

    rc_curve = wandb.plot.roc_curve(targets, preds, labels=labels)
    wandb_run.log({plot_name: rc_curve})

def wandb_p_r_curve(wandb_run, targets, preds, labels=None, plot_name="pr_curve"):
    """
    The plot_name represents the id of the precision-recall curve on wandb. Do NOT change it
    over the runs.
    """

    pr_curve = wandb.plot.pr_curve(targets, preds, labels=labels)
    wandb_run.log({plot_name: pr_curve})

def wandb_confusion_matrix(wandb_run, targets, preds, labels=None, plot_name='single_run_cm'):
    """
    Compute a scikit-learn confusion matrix, store it and log it on wandb (if using wandb)
    """
    cm = metrics.confusion_matrix(targets, preds, labels=labels)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Plot the confusion matrix
    disp.plot(cmap='Blues', values_format='d')
    wandb_run.log({plot_name, plt})
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')

    # Save the plot to a file
    #plt.savefig('confusion_matrix.png')
    plt.close()  # Close the figure to free up memory

def wandb_multi_run_confusion_matrix():
    # À faire
    pass


