def train(num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Resnet()
    model.to(device)
    model.train()
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 0.0001)
    best_accuracy = 0.0
    print("Begin training...") 
    num_bitch = 0
    for epoch in range(1, num_epochs+1): 
        running_train_loss =0.0
        running_accuracy = 0.0
        running_vall_loss = 0.0
        total = 0.0
        model.train()  
        # Training Loop 
        for data in train_loader: 
        #for data in enumerate(train_loader, 0): 
            inputs, outputs = data  # get the input and real species as outputs; data is a list of [inputs, outputs] 
            optimizer.zero_grad()   # zero the parameter gradients          
            predicted_outputs = model(inputs.to(device))   # predict output from the model 
            train_loss = loss_fn(predicted_outputs.to(device), outputs.to(device))   # calculate loss for the predicted output  
            train_loss.backward()   # backpropagate the loss 
            optimizer.step()        # adjust parameters based on the calculated gradients 
            running_train_loss +=train_loss.item()  # track the loss value
            num_bitch += 1 
            if (num_bitch % 100) == 0 :
                print(num_bitch)

        # Calculate training loss value 
        train_loss_value = running_train_loss/len(train_loader) 
        model.eval()

        # Validation Loop 
        with torch.no_grad(): 
            for data in validate_loader: 
               inputs, outputs = data 
               inputs = inputs.to(device)
               outputs = outputs.to(device)
               predicted_outputs = model(inputs) 
               val_loss = loss_fn(predicted_outputs, outputs) 
             
               # The label with the highest value will be our prediction 
               _, predicted = torch.max(predicted_outputs, 1)
               running_vall_loss += val_loss.item()  
               total += outputs.size(0) 
               running_accuracy += ((predicted == outputs).sum().item())
 
        # Calculate validation loss value 
        val_loss_value = running_vall_loss/len(validate_loader) 
                
        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.  
        accuracy = (100 * running_accuracy / total)     
 
        # Save the model if the accuracy is the best 
        if accuracy > best_accuracy: 
           torch.save(model.state_dict(),'/content/drive/MyDrive/Colab Notebooks/cow/model.pt') 
           best_accuracy = accuracy 
         
        # Print the statistics of the epoch 
        print('Completed training batch', epoch, 'Training Loss is: %.4f' %train_loss_value, 'Validation Loss is: %.4f' %val_loss_value, 'Accuracy is %d %%' % (accuracy))

train(50)
