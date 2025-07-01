# Resources

Resources are the dynamic or static feedback and provisioning system used by the Workflow Forge system in order to allow placeholders to be filled. Resources have two parts to them. These are **Resource Specifications** and **Resources** themselves

## Resource Specifications and Resources

Resource specifications are contracts that can be formed in the UDPL system to connect to a resource later on. They are declaration of intent, including the parameters to call with, the resource to invoke, and where in the compilation chain we expect this to be resolved with. For the most up-to-date material you should always check [UDPL](UDPL.md).

Resources, on the other hand, fulfill these specifications. They are provided as a Dict[str, Any] mapping, where the abstract resource can be one of many times, during the python stage of compiling. They also may at times be even more restricted.

### When a Resource Specification is Required.

The need for a resource specification is established in UDPL any time a placeholder is defined in the text. The following, for instance, demands we create a valid resource specification or the UDPL file will throw an error when parsing.

```toml
[[solving]]
text = """
[Prompt] Consider this feedback: {feedback} #<-This line define that a resource is needed
[Answer]
"""
tags = [["Training"], ["Feedback"]]
```

### Resource Specification

A resource specification is a toml element in terms of {sequence_name}.{placeholder_name}. It has three arguments, two of which are optional. These are

* **name**: Required. The name of the resource to look for, and technically the name we expect it to be called at in the dictionary later on.
* **type**: The resource type. Optional. Options are 'standard', 'custom', and 'argument'
* **arguments**: Keyword arguments. A dictionary. These will be fed into the resource as kwargs. Not providing it indicates there are no arguments.

An example of a few different valid specifications are shown below. The first simply invokes a resource named feedback with no parameters, and this resource would have to be provided 

```toml
[solving.feedback]
name = "feedback_sampler"
arguments = { num_samples = 3 }
type = "custom"
```

## Resources

Resources are, ultimately, anything that will compile to the AbstractResource class. AbstractResource is found under workflow_forge.frontend.resources, as are all the other resource types. Resources are provided in terms of a Dict[str, AbstractResource] dictionary, or possibly Dict[str, Any] and compiled to that type, depending on stage. A resource is, basically, a promise to return a string when invoked, possibly with arguments.

Resources are sampled at the time the workflow factory is run. This means that the same workflow factory can be updated between runs by updating the underlying resources.

### Kinds of resources

resources are found under workflow_forge.resources. At the time of this writing, there are the following. This may have since been updated, so it is recommended to check the resources file yourself to be sure.

* **AbstractResource** :The base contract. It is a class that can be called with kwargs and returns a string. It is also usually what you subclass to make your own resources.
* **StaticStringResource** It is constructed with a string, then invoked with no kwargs and returns that string. Most python types are compiled to this, but more advance sampling can be used in, for example, constiutional workloads
* **ListSamplerResource**: Constructed with a list of strings, this will sample from among them. It requires a kwarg "num_samples" that resolves to an integer, retrieves that many samples randomly without replacement, clamping if you ask for too many. It then concatenates the samples together using "\n" and return.
* **LRUBufferResource**: A specialized resource which can be appended with strings and maintains an internal LRUBuffer to sample from. Excellent for feedback systems. Uses the same underlying logic as ListSamplerResource, and is indeed a subclass of it.

## When resources are sampled and errors raise

Resources are sampled when you run the workflow factory function located above. This means that certain kinds of errors - notably providing the wrong arguments - may not be detected until this point. However, generally we fail as early as we can.


### How to store resources

Resources should be stored in a dictionary that maps the "name" feature of the resource specification
onto the resource you intend to use to resolve it. For instance, the below dictionary prepares to resolve statement_a, statement_b

```python
import workflow_forge as forge

resources = {"statement_a" : forge.resources.StaticStringResource("This is statement a"),
             "statemement_b" : forge.resources.StaticStringResource("This is statement b")
             }
```

All resources, at all possible injection stages, are passed around in terms of dictionaries. It should be noted, however, that depending on the stage it may sometimes be possible to do something like.

```python
resources = {
             "statement_a" : "This is statement a",
             "statement_b" : "This is statement b"
            }
```


### When you can provide resources

You can provide resources at three stages during compilation, though being able to do so depends on setting the right type. 

**Program Setup Phase**

You may provide resources during the program setup phase. This is the standard kind of resource. **Note that unless you provide a type, the resource must be defined here, and this is what standard means**. Note additionally that these cannot be python types, but must be an AbstractResource subclass.

```toml
[setup.constitution_detauls]
name = "constitution_sampler"
arguments = { num_samples = 3 }
type = "standard"
```

```python
import workflow_forge as forge

...the stuff in the meantime
constitution_details = ...
resources = {"constitution_sampler" : forge.resources.ListSamplerResource(constitution_details)}

program = forge.new_program(sequences, resources, config)
```

This is now configured to sample three entries from the constitution list. This is the Cannonical way to hook up constitutions, feedback systems, and other permanent connective systems.

**Sequence Setup Phase**

Python types that can be converted to a string may be defined when you are invoking a sequence. This is extremely useful particularly when defining flow control, or other features where you really want to understand what is being specified in the python code itself. An example of this is shown below. We can set up the loop statement as this:

```toml
[[loop]]
text = """
[Prompt] You are in a loop control statement. If this is your first time seeing it, say
nothing. If you would like to rethink your answer, say nothing. If you are satisfied with your
answer, say [Escape] "[Jump]" [EndEscape]. Loop at least {min} and at most {max} times.
[Answer]
"""
tags = [[], []]

# Notice we MUST set the type to 'custom' or above it to 'argument' or this will throw.
# Externally provided details also should not have arguments.
[loop.min]
name = "min"
type = 'custom' # or 'argument', but narrow typing is good.

# Notice we must define both resources.
[loop.max]
name = "max"
type = 'custom'

```

Then later on in our program pass in min and max

```python
# Create subroutine
subroutine = forge.new_program(sequences, resources, config)
with subroutine.loop("loop", min=3, max=6) as loop_scope:
    loop_scope.run("think")
```

This should be considered the Canonical way to configure flow control details, as it is considerably clearer and more flexible than statically defining these numbers in the UDPL system. However, in theory any such details cna be passed along this way. As you can see, this will form the backend resources dictionary automatically out of the extra keyword arguments, and then resolve them to StaticStringResource.

**Workflow Sampling Phase**

Finally, resources can be provided as a dictionary of resources or python types when constructing the workflow itself. This should generally only be used for passing along things that change each workflow, such as user input. An example might be the following

```toml
[[prompt]]
text ="""
[Prompt] The user is asking the following of you: {prompt}
[Answer]
"""
tags = [[], ["response"]]

[prompt.prompt]
name = "prompt"
type = "argument"
```

We can now perform our construction up to the workflow factory, and do

```python
workflow_factory = program.compile()
workflow = workflow_factory(prompt="What is the maximum speed of a bumblebee?")
```

This should be considered the Canonical way to pass in user input. It can also, of course, be used for other inputs but you would likely be better served using a resource or even a custom resources

## Making custom resources

Resources can easily be customized for whatever purpose you need them to be. The system is designed to be extended. For instance, lets say you are dissatisfied with the standard ListSamplerResource as you want Random sampling not random with no replacement. The abstract class you need to subclass is located in workflow_forge.resources as AbstractResource, and has this specification

```python
class AbstractResource(ABC):
    """
    The abstract resource class. Specifies
    the usage contract
    """
    @abstractmethod
    def __call__(self, **kwargs)->str:
        """
        Must return a string of some sort to resolve
        the dynamic dependency
        :param kwargs: The keyword arguments
        :return: A string
        """
```

All that you need to do is implement this. So you spend a little bit thinking, and quickly come up with:

```python
import workflow_forge as forge
import numpy as np
from workflow_forge.resources import AbstractResource
from typing import List, Union


class MyCustomResource(AbstractResource):
    """
    A string sampler resource is capable of drawing
    samples randomly from among an internal
    list of strings.
    """

    def init(self, string_list: List[str]):
        assert len(string_list) > 0
        self.string_list = string_list

    def call(self, num_samples: Union[int, str]) -> str:
        output = []
        if isinstance(num_samples, int):
            samples = np.random.randint(0, len(self.string_list), size=num_samples)
            for idx in samples:
                output.append(self.string_list[idx])
        elif isinstance(num_samples, str):
            if num_samples == "all":
                output = self.string_list
            else:
                raise NotImplementedError("Unknown string sampling type")
        return "\n".join(output)
```

Now, you can setup a resource spec to use it that passes in the number of things to sample. Other resource specifications can be used in much the same way, with the only requirement being the arguments match up

```toml

[[constitution]]
text = """
[Prompt] You are operating under a constitution, and want to generally conform to it. A few 
of the principles are {constitution_samples}
[Answer] I understand [EOS]
"""
tags = [[], []]

[constitution.constitution_samples]
name = "constitution_samples"
arguments = {"num_samples" = 10} # <- must match argument kwargs you defined.
```
