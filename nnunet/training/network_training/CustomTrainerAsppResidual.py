from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.network_architecture.CustomTrainerAsppResidual import Custom3DUNet

class CustomTrainerASPPResidual(nnUNetTrainerV2):
    def initialize_network(self):
        """ Use the custom 3D UNet with ASPP and residual connections """
        self.network = Custom3DUNet(self.num_input_channels, self.base_num_features, 
                                    self.num_classes, self.net_num_pool_op_kernel_sizes)
        self.network.to(self.device)
