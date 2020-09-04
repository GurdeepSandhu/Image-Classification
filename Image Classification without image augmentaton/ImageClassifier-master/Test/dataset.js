 class dataset {

    constructor(numClasses) {
        this.numClasses = numClasses;
        this.labels = []; 

        this.samples = {
        xs: null,
        ys: null
        }
    }

    addSample(sample, label) {
        let labelID;
        if(this.labels.includes(label)) {
            labelID = this.labels.indexOf(label);
        } else {
            labelID = this.labels.push(label) -1;
        }
        console.log("LabelID" + labelID);
        console.log("LabelsArray" + this.labels);

        //Representing a probability distribution using one-hot encode label
        const y = tf.tidy(
            () => tf.oneHot(tf.tensor1d([labelID]).toInt(), this.numClasses))
        

        // adding details to training data and using tf.keep to prevent tf.tidy from being disposed
            if(this.samples.xs == null) 
                {
                    this.samples.xs = tf.keep(sample);
                    this.samples.ys = tf.keep(y);
                } else {
                    const xs = this.samples.xs;
                    const ys = this.samples.ys; 
                    this.samples.xs = tf.keep(xs.concat(sample, 0));
                    this.samples.ys = tf.keep(ys.concat(y,0));
                    xs.dispose(); 
                    ys.dispose();
                    y.dispose(); 
                }
            
            
    }

     getXS() {
        return this.samples.xs; 
    }

    getYS() {
        return this.samples.ys; 
    }
    
    getLabelLength(){
        console.log(this.labels.length); 
        return this.labels.length; 
    }
    getLabel() {
        return this.labels; 
    }


}