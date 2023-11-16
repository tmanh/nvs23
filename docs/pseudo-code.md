# PSEUDO-CODE

## Pseudo-code for the `training` method (Python-style)

```python
def forward(
    self,
    input_imgs, output_imgs, raw_depth, augmented_depth,
    K, src_RTs, dst_RTs
    ):

    # Initialize depth_loss to 0
    depth_loss = 0
    
    # Apply depth completion and compute depth loss
    completed_depth, estimated_depth, visual_features, spatial_features = self.depth_completion(
        augmented_depth, input_imgs
    )
    depth_complete_loss, depth_estimate_loss = self.compute_depth_loss(
        raw_depth, completed_depth, estimated_depth
    )

    # Perform rendering
    prj_visual_features, prj_spatial_features = self.rendering(
        K, src_RTs, dst_RTs,
        visual_features, spatial_features, completed
    )

    # Compute enhanced images and loss
    synthesized_image = self.apply_fusion_network(
        prj_visual_features, prj_spatial_features
    )
    visual_loss = self.compute_visual_loss(synthesized_image, output_imgs)

    # Add depth loss to the loss function
    loss = {
        "depth_complete_loss": depth_complete_loss,
        "depth_estimate_loss": depth_estimate_loss
        "visual_loss": visual_loss
    }

    return loss
```

## Pseudo-code for the `depth_completion` method (Python-style)

```python
def forward(self, color, depth):
    swin_feats = self.swin_transformer(color)
    vgg_feats = self.vgg16_shallow(color)
    visual_feats = self.combine_visual_features(vgg_feats, swin_feats)

    color_spatial_features = self.color_spatial_decoder(visual_feats)
    estimated = self.depth_from_color_conv(color_spatial_features)
    
    depth_spatial_features = self.depth_encoder(depth)

    spatial_features = self.fusion_decoder(
        color_spatial_features, depth_spatial_features
    )
    completed = self.depth_from_merge_conv(spatial_features[-1])
    
    spatial_features = self.combine_spatial_features(spatial_features, completed)
    visual_features = self.combine_visual_features(visual_features, color)

    return completed, estimated, visual_feats, spatial_features
```

## Pseudo-code for the `rendering` method (Python-style)

```python
def rendering(
    self, K, src_RTs, dst_RTs,
    visual_features, spatial_features, completed
    ):
    num_inputs = self.input_view_num
    num_outputs = dst_RTs.shape[1]

    prj_visual_features = []
    prj_spatial_features = []

    for i in range(num_inputs):
        pts_3D_nv = self.pts_transformer.view_to_world_coord(
            completed[:, i], K, src_RTs[:, i])

        src_visual_feats = visual_features[:, i:i + 1]
        src_spatial_feats = visual_features[:, i:i + 1]

        pointcloud = self.pts_transformer.world_to_view(
                pts_3D_nv, K, dst_RTs,
        )

        prj_visual_feats, prj_spatial_feats = self.pts_transformer.splatter(
            pointcloud, src_visual_feats, src_spatial_feats, depth=True)

        prj_visual_features.append(prj_visual_feats)
        prj_spatial_features.append(prj_spatial_feats)

        return torch.stack(prj_visual_feats, 0), torch.stack(prj_spatial_feats, 0)
```

## Pseudo-code for the `fusion` method (Python-style)

```python
def apply_fusion_network(
        prj_visual_features, prj_spatial_features
    ):
    c_hs = None
    d_hs = None
    out_colors = []
    alphas = []
    for vidx in range(n_views):
        y, c_hs, d_hs = self.merge_net(
            prj_visual_features[:, vidx], prj_spatial_features[:, vidx],
            c_hs, d_hs
        )
        self.estimate_view_color(y, out_colors, alphas)

    return self.compute_out_color(out_colors, alphas)


def estimate_view_color(self, x, out_colors, alphas):
    out_colors.append(self.rgb_conv(x))
    alphas.append(self.alpha_conv(x))

def compute_out_color(self, colors, alphas):
    colors = torch.stack(colors)
    alphas = torch.stack(alphas)

    alphas = torch.softmax(alphas, dim=0)
    return (alphas * colors).sum(dim=0)
```

## Pseudo-code for the `forward` method of merge_net (Python-style)

```python
def forward(self, c, d, c_hs=None, d_hs=None):
    if c_hs is None:
        c_hs = [None for _ in range(self.n_rnn)]
        d_hs = [None for _ in range(self.n_rnn)]

    c, feats, hidx = self.encode(c, d, c_hs, d_hs)
    c, c_hs = self.decode(c, c_hs, feats, hidx)
        
    return c, c_hs, d_hs

def encode(self, c, d, c_hs, d_hs):
    c_hidx = 0
    c_feats = []
    for c_enc, d_enc in zip(self.encoders, self.d_encoders):
        for c_mod, d_mod in zip(c_enc, d_enc):
            d = d_mod(d)
            if isinstance(c_mod, self.enc_gru_conv):
                c = c_mod(c, d, c_hs[c_hidx], d_hs[c_hidx])
                c_hs[c_hidx] = c
                d_hs[c_hidx] = d
                c_hidx += 1
            else:
                c = c_mod(c)
        c_feats.append(c)

    return c, c_feats, c_hidx

def decode(self, x, hs, feats, hidx):
    for dec in self.decs:
        x0 = feats.pop()
        x1 = feats.pop()
        x0 = functional.interpolate(
            x0, size=(x1.shape[2], x1.shape[3]), mode='nearest'
        )
        x = torch.cat((x0, x1), dim=1)
        for mod in dec:
            if isinstance(mod, (self.dec_gru_conv, self.last_conv)):
                x = mod(x, hs[hidx])
                hs[hidx] = x
                hidx += 1
            else:
                x = mod(x)
        feats.append(x)
        x = feats.pop()

    return x, hs
```