
## Build MOdel

import torch
import torch.nn as nn
from Utils import seeding

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, device=None):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, device=device),
            nn.BatchNorm2d(out_c, device=device),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c, device=None):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0, device=device)

    def forward(self, x):
        return self.deconv(x)


#######################UNETR_2D_CD#######################
#######################UNETR_2D_CD#######################

class UNETR_2D_CD(nn.Module):
    def __init__(self, cf, device=None):
        super().__init__()
        self.cf = cf
        self.device = device

        """ Patch + Position Embeddings """
        self.patch_embed = nn.Linear(
            cf["patch_size"]*cf["patch_size"]*cf["num_channels"],
            cf["hidden_dim"],
            device=self.device
        )

        self.positions = torch.arange(start=0, end=cf["num_patches"], step=1, dtype=torch.int32, device=self.device)
        self.pos_embed = nn.Embedding(cf["num_patches"], cf["hidden_dim"], device=self.device) 
        # self.pos_embed = nn.Parameter(torch.randn(1, cf["num_patches"], cf["hidden_dim"]))

        """ Transformer Encoder """
        self.trans_encoder_layers = []

        for i in range(cf["num_layers"]):
            layer = nn.TransformerEncoderLayer(
                d_model=cf["hidden_dim"],
                nhead=cf["num_heads"],
                dim_feedforward=cf["mlp_dim"],
                dropout=cf["dropout_rate"],
                activation=nn.GELU(),
                batch_first=True,
                device=self.device
            )
            self.trans_encoder_layers.append(layer)


        """ CNN Decoder """
        ## Decoder 1
        self.d1 = DeconvBlock(cf["hidden_dim"], 512, device=self.device)
        self.s1 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 512, device=self.device),
            ConvBlock(512, 512, device=self.device)
        )
        self.c1 = nn.Sequential(
            ConvBlock(512+512, 512, device=self.device),
            ConvBlock(512, 512, device=self.device)
        )

        ## Decoder 2
        self.d2 = DeconvBlock(512, 256, device=self.device)
        self.s2 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 256, device=self.device),
            ConvBlock(256, 256, device=self.device),
            DeconvBlock(256, 256, device=self.device),
            ConvBlock(256, 256, device=self.device)
        )
        self.c2 = nn.Sequential(
            ConvBlock(256+256, 256, device=self.device),
            ConvBlock(256, 256, device=self.device)
        )

        ## Decoder 3
        self.d3 = DeconvBlock(256, 128, device=self.device)
        self.s3 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 128, device=self.device),
            ConvBlock(128, 128, device=self.device),
            DeconvBlock(128, 128, device=self.device),
            ConvBlock(128, 128, device=self.device),
            DeconvBlock(128, 128, device=self.device),
            ConvBlock(128, 128, device=self.device)
        )
        self.c3 = nn.Sequential(
            ConvBlock(128+128, 128, device=self.device),
            ConvBlock(128, 128, device=self.device)
        )

        ## Decoder 4
        self.d4 = DeconvBlock(128, 64, device=self.device)
        self.s4 = nn.Sequential(
            ConvBlock(3, 64, device=self.device),
            ConvBlock(64, 64, device=self.device)
        )
        self.c4 = nn.Sequential(
            ConvBlock(64+64, 64, device=self.device),
            ConvBlock(64, 64, device=self.device)
        )

        """ Output """
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0, device=self.device)
        self.out_sigmoid = nn.Sigmoid()

    def to(self, device):
        super().to(device)
        self.device = device  # Store device when moved
        return self

################FORWARD PASS################
################FORWARD PASS################

    def forward(self, m, n):
        """ Patch + Position Embeddings """
        print (f'm device is: {m.device}')
        print (f'n device is: {n.device}')
        print ("m-shape: " , m.shape, "n-shape", n.shape)
        
        patch_embed1 = self.patch_embed(m).to(self.device)   ## [8, 256, 768]
        print ("patch_embed1 shape: " , patch_embed1.shape, "patch_embed1 device: ", patch_embed1.device)
        
        positions = self.positions.to(self.device) 
        print ("positions shape: " , positions.shape, "positions device: ", positions.device)
        
        pos_embed = self.pos_embed(positions).to(self.device)    ## [256, 768]
        print ("pos_embed shape: " , pos_embed.shape, "pos_embed device: ", pos_embed.device)
        # pos_embed = self.pos_embed
        

        x = patch_embed1 + pos_embed ## [8, 256, 768]
        print ("x shape: " , x.shape, "x device: ", x.device)
        x = x.to(self.device)

        patch_embed2 = self.patch_embed(n).to(self.device)    ## [8, 256, 768]
        y = patch_embed2 + pos_embed ## [8, 256, 768]
        y = y.to(self.device)

        """ Transformer Encoder """
        skip_connection_index = [3, 6, 9, 12]
        skip_connections1 = []
        skip_connections2 = []

        for i in range(self.cf["num_layers"]):
            layer = self.trans_encoder_layers[i]
            x = layer(x)

            if (i+1) in skip_connection_index:
                skip_connections1.append(x)

        for j in range(self.cf["num_layers"]):
            layer = self.trans_encoder_layers[j]
            y = layer(y)

            if (j+1) in skip_connection_index:
                skip_connections2.append(y)

        """ CNN Decoder """
        x_z3, x_z6, x_z9, x_z12 = skip_connections1
        y_z3, y_z6, y_z9, y_z12 = skip_connections2

        ## Reshaping
        batch = m.shape[0]
        x_z0 = m.view((batch, self.cf["num_channels"], self.cf["image_size"], self.cf["image_size"]))
        print ("x_z0 shape: " , x_z0.shape, "x_z0 device: ", x_z0.device)
        y_z0 = n.view((batch, self.cf["num_channels"], self.cf["image_size"], self.cf["image_size"]))

        shape = (batch, self.cf["hidden_dim"], self.cf["patch_size"], self.cf["patch_size"])
        x_z3 = x_z3.view(shape)
        x_z6 = x_z6.view(shape)
        x_z9 = x_z9.view(shape)
        x_z12 = x_z12.view(shape)

        y_z3 = y_z3.view(shape)
        y_z6 = y_z6.view(shape)
        y_z9 = y_z9.view(shape)
        y_z12 = y_z12.view(shape)

        ## Decoder 1
        x = self.d1(x_z12)
        print ("x shape: " , x.shape, "x device: ", x.device)
        s = self.s1(torch.abs(x_z9-y_z9))
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)

        ## Decoder 2
        x = self.d2(x)
        s = self.s2(torch.abs(x_z6-y_z6))
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)

        ## Decoder 3
        x = self.d3(x)
        s = self.s3(torch.abs(x_z3-y_z3))
        x = torch.cat([x, s], dim=1)
        x = self.c3(x)

        ## Decoder 4
        x = self.d4(x)
        s1 = self.s4(x_z0)
        s2 = self.s4(y_z0)
        s = torch.abs(s1-s2)
        x = torch.cat([x, s], dim=1)
        x = self.c4(x)

        """ Output """
        output = self.output(x)
        output = self.out_sigmoid(output)

        return output
    

if __name__ == "__main__":

    seeding(42)

    config = {}
    config["image_size"] = 256
    config["num_layers"] = 12
    config["hidden_dim"] = 768
    config["mlp_dim"] = 3072
    config["num_heads"] = 12
    config["dropout_rate"] = 0.1
    config["num_patches"] = 256
    config["patch_size"] = 16
    config["num_channels"] = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = torch.rand((8, 256, 768), dtype=torch.float32, device=device)
    n = torch.rand((8, 256, 768), dtype=torch.float32, device=device)
    
    f = UNETR_2D_CD(config, device=device)
    out = f(m,n)

    print("MODEL OUTPUT SHAPE",out.shape, "MODEL OUTPUT TYPE", out.dtype, "MODEL OUTPUT DEVICE", out.device)

     # Get the minimum value in the tensor
    min_value = torch.min(out)
    print(f"Out Minimum value: {min_value}")
    # Get the maximum value in the tensor
    max_value = torch.max(out)
    print(f"Out Maximum value: {max_value}")