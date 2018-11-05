# Proposal Requirements
*9.523 | Fall 2018*

The project proposal should be submitted through Stellar. The deadline is on
Nov. 9th, but feel free to submit it as soon as you have it. You can get
feedback and get started with the project earlier if you want.

## Tentative Timeline

-  [ ] Week 0, Nov 5 - 9: Prepare proposal and write out pseudocode.
-  [ ] Week 1, Nov 12 - 16: Background, identify what's already been done and   
   find existing tools that we can use.     
-  [ ] Week 1 & 2, Nov 12 - 23: Set up the framework for the implementation and
   generate first version of the project code.
-  [ ] Week 2 & 3, Nov 19 - 30: Iterate on the implementation based on
   preliminary results.


## The project proposal should contain:

* Title: __Recapitulation of Functional Connectomes from a Simulated Cortical
  Hypercolumn__

* Team members:
  * __Corban Swain__
  * __Nick Ning__

* Abstract:

  1. One or two sentences providing a basic introduction to the field.
     *  __Physiological recording of brain activity in organisms from *C.
        elegans* to *Homo sapiens* is one of the primary methods for furthering
        our understanding of the brain.__
     *  __Part of this "understanding" lies in the ability to elucidate the
        topology of effective connections between neurons--i.e. building brain
        networks (Bullmore, 2009).__
     *  __A better understanding of brain networks can be applied to build     
        more realistic models neuronal computations and generate more
        physiologically-motivated framework for diseases of the brain.__

  2. Two to three sentences of more detailed background and previous work.
     *  __Existing techniques for recording brain activity span a range of
        temporal and spatial resolving abilities (see Table 1 for a limited
        summary of these methods).__
     *  __In the context of this proposal, we give attention to opportunities
        provided by recent developments in genetically-expressed fluorescent
        ion and voltage reporters: a technology which, in combination with
        volumetric optical imaging, can be used to record from individual
        neurons across large regions of the brain in transparent organisms like
        zebrafish.__
     *  __Existing techniques for generating a network graphs from brain    
        recordings are built for recordings from electrodes and fMRI and rely
        primarily on correlative techniques and produce undirected
        connectivity models.__

  3. One sentence clearly stating the general problem being addressed.
     *  __In this proposal we will begin to address gap in existing techniques
        to generate directional graphs of neuron connections from functional
        recordings of brain activity and to effectively harness the power of
        large volume, near single-cell recordings made possible by fluorescent
        reporters with optical imaging.__

  4. One sentence summarizing the expected main result (enumerate
     multiple possibilities).  
     *  __We expect to demonstrate in silico that a recurrent neural network   
        (RNN) can be trained using activity recordings from many small known
        networks to accurately predict the directional graph of an unobserved
        network. We might find that RNNs are also capable of predicting the
        input to a simulated unobserved network. Additionally, we have
        considered the possibility that different machine learning
        architectures would each be more accurate at predicting different
        aspects of the network topology.__

  5. Two or three sentences explaining what the main result is
     expected to reveal in direct comparison to previous work.
     *  __The successful completion of this work would provide a novel    
        computational tool for the analysis of functional brain recordings.__
     *  __Specifically, we expect our proposed tool to be able to to use
        fluorescence-based recordings (after image preprocessing) to infer
        functional connectomes with directional information ((need
        justification)). We also expect that our tool will be well posed,
        unlike correlative methods, to integrate the prediction of the synaptic
        properties (e.g. reeptor type ((i need better exaples here))) of neuron
        connections because of its ability to train on recordings annotated
        with information beyond its functional connections.__

* Lecture/s related to the project: explain how the project relates to one or
  more lectures from 9.523 (two sentences per related lecture)
   * __lec 3 - Hippocampal mechanisms of memory and cognition__
   * __lec 7 - Functional Modules: what good are they and how do we
     get them?__
   * __lec 9 - (something goes here)__

* Methods: Plan of the experiment/s to execute during the project. Detail
  computational resources needed or any material needed for the psychophysics
  experiments.
  1.  __Model of Cortial Hypercolumn [[Hansel 1998]]__


## Brainstorming

*  Original Pitch - I am interested in working on simulation of temporal and
   phase encoding for learning and information transmission within a learning
   network. Very open to other ideas as well.
*  The Goal - __The ability to predict network connectivity from individual
   neuron recordings.__
*  The Method - __To use the simulation of a simple network to__
*  Neuron Recordings - The use of probes to detect and record the activity of
   neuronal cells over time. There are many technologies which can be used to detect the activity of neurons.
   *  Patch clamps can directly measure the difference in electrical potential
      (i.e. voltage) between the inside and out side of a neuronal cell.
   *  Electrodes can measure the electrical potential fields emanating from
      current-conducting neurons.
   *  Genetically expressed fluorescent reporters (e.g.)
*  Have a logical flow to the paper ... address a somewhat relevant.
*  __We propose applying machine learning to both predict the state and
  connections of a small, simulated network of spiking neurons.__


## Discard Pile
*  __There exist many methods to record activity from the brain of__
*  __Importance of constructing network connectivity from activity
   data/recordings.__
*  __Explanation of old methods for inferring connectivity from activity
   time courses.__
