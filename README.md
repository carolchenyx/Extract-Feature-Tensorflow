# Extract-Feature-Tensorflow
Extract feature from each layer of neural network in tensorflow

### There are many bugs when you want to extract the feature from each layer while training the neural network. I used two days to solve this problem. In order to help the people who want to do this, I will write the solution I used in here. Hope to help you.


# First step: save each layer's name. For example,
    
    self.layers = []
    
    #this is one of the layer of my neural netwok
    
    output = lProvider.inverted_bottleneck(output, t, output_channel, stride, k_s=3, dilation=1, scope=layerDescription)
    
    #After this layer, I used a list to append the name of this layer. The code show below:
    
    self.layers.append(output.op.name)
 
# Second step: check your sess.run() structure in training part. Take my code as an example:
    #This is the example of my code,expecially for the feed_dict, three parts in it.
    
    res = self.sess.run([self.output, self.summaryMerge],
          feed_dict={self.inputImage: inputs, self.heatmapGT: heatmaps, self.learningRate: 0})

# Third step: write the extraction code 
     # self.layer includes all layers' name
     
     for l in range(len(self.layers)):
          featureout = self.sess.graph.get_tensor_by_name("{}:0".format(self.layers[l]))
          feature = self.sess.run([featureout,self.summaryMerge.graph.get_operation_by_name(self.layers[l])],
                          feed_dict={self.inputImage: inputs, self.heatmapGT: heatmaps, self.learningRate: 0})
                          
     #"{}:0".format(self.layers[l]): to get the name of the layer, "0" is important. If you don't write that, the sess.run can not find the layer.
     #self.sess.graph.get_tensor_by_name("{}:0".format(self.layers[l])): the whole structure from the beginning to the layer which the name you have given
     #self.inputImage: inputs, self.heatmapGT: heatmaps, self.learningRate: 0}:the format according to my traning part.
     
## If you have any questions, welcome to discuss with me.

                    
   
