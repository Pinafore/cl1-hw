
import unittest
from lorabert_buzzer import LoRALayer, LinearLoRA, initialize_base_model, add_lora
import torch
from torch import Tensor

class LoraBertTest(unittest.TestCase):
    def setUp(self):
        self.in_dim = 3
        self.out_dim = 4
        self.rank = 2
        self.alpha = 0.5

        #all possible modules we could adapt for each layer
        self.all_modules = {"output": ["dense"], "intermediate": ["dense"]}
        #modules to adapt for each layer
        self.modules = [{"intermediate": ["dense"]},
                        {"intermediate": ["dense"], "output": ["dense"]},
                        {"output": ["dense"]}]

        self.bert = [initialize_base_model(model_name="hf-internal-testing/tiny-random-BertForSequenceClassification")
                     for _ in self.modules]

        self.lora_layer = LoRALayer(in_dim=self.in_dim, out_dim=self.out_dim, rank=self.rank, alpha=self.alpha)

    def util_tensor_compare(self, model: Tensor, reference: Tensor, extra_info: str=""):
        for position, test_result in enumerate(torch.isclose(model, reference)):
            self.assertTrue(test_result, "Position %i: %0.2f != %0.2f (result=%s, reference=%s)" %
                                (position, model[position], reference[position], str(model), str(reference)) +
                                ": %s" % extra_info if extra_info else "")

    def test_lora_dim(self):
        self.assertEqual(self.lora_layer.A.shape, (self.in_dim, self.rank))
        self.assertEqual(self.lora_layer.B.shape, (self.rank, self.out_dim))


    def test_gradient_flow(self):
        # Ensure gradients flow correctly through LoRA parameters

        x = torch.FloatTensor([2, 4, 6]).requires_grad_()

        y = self.lora_layer(x)

        y.sum().backward()

        self.assertIsNotNone(x.grad, "Gradients should flow back to input")
        self.assertTrue(torch.any(self.lora_layer.A.grad != 0) or torch.any(self.lora_layer.B.grad != 0), "Gradients should flow to LoRA layers matrix")

    def test_lora_forward(self):
        with torch.no_grad():
            self.lora_layer.A.data = torch.FloatTensor([[1, 0], [0, 3], [0, 0]])
            self.lora_layer.B.data = torch.FloatTensor([[1, 0, 4, 6], [0, 3, -1, 3]])

        x = torch.FloatTensor([2, 4, 6])

        self.util_tensor_compare(self.lora_layer(x), torch.FloatTensor([ 1., 18., -2., 24.]), "lora forward")

    def test_bert_reparameterize(self):
        
        for ii, module_config in enumerate(self.modules):
            model, tokenizer = self.bert[ii]

            # Before we adapt layers, check that all modules are Linear layers
            for layer in model.bert.encoder.layer:
                for component_name in self.all_modules:
                    component = getattr(layer, component_name)
                    for module_name in self.all_modules[component_name]:
                        module = getattr(component, module_name)
                        self.assertIsInstance(module, torch.nn.Linear, "Model %i pretest: Module %s.%s is not a Linear layer; Config: %s" % (ii, component_name, module_name, str(module_config)))

            add_lora(model.bert.encoder, rank=self.rank, alpha=self.alpha, modules_to_adapt=module_config)

            for layer in model.bert.encoder.layer:
                #check that all modules we adapted are LoRA layers
                for component_name in module_config:
                    component = getattr(layer, component_name)
                    for module_name in module_config[component_name]:
                        module = getattr(component, module_name)
                        self.assertIsInstance(module, LinearLoRA, "Model %i posttest: Module %s.%s is not a LoRA layer; Config: %s" % (ii, component_name, module_name, str(module_config)))

                #check that all other modules are not LoRA layers
                for component_name in self.all_modules:
                    if not component_name in module_config:
                        component = getattr(layer, component_name)
                        for module_name in self.all_modules[component_name]:
                            module = getattr(component, module_name)
                            self.assertIsInstance(module, torch.nn.Linear, "Model %i posttest: Module %s.%s is not a Linear layer; Config: %s" % (ii, component_name, module_name, str(module_config)))


            


if __name__ == '__main__':
    import logging
    unittest.main()
