def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def set_trainable_params_for_models_1(model):
    freeze_model(model)
    for name, param in model.named_parameters():
        if "auto_model" not in name:
            param.requires_grad = True

def set_trainable_params_for_models_2(model):
    freeze_model(model)
    for name, param in model.named_parameters():
        if "project_1" in name:
            param.requires_grad = True

def set_trainable_params_for_models_3(model):
    freeze_model(model)
    for name, param in model.named_parameters():
        if "project_2" in name or "control_model" in name:
            param.requires_grad = True