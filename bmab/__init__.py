from bmab import nodes


NODE_CLASS_MAPPINGS = {
    'Basic': nodes.BMABBasic,
    'Edge': nodes.BMABEdge,
    'Integrator': nodes.BMABIntegrator,
    'Extractor': nodes.BMABExtractor,
    'SeedGenerator': nodes.BMABSeedGenerator,
    'BMAB KSampler': nodes.BMABKSampler,
    'BMAB KSamplerHiresFix': nodes.BMABKSamplerHiresFix,
    'BMAB Upscaler': nodes.BMABUpscale,
    'BMAB Face Detailer': nodes.BMABFaceDetailer,
    'BMAB Person Detailer': nodes.BMABPersonDetailer,
    'BMAB Simple Hand Detailer': nodes.BMABSimpleHandDetailer,
    'BMAB Subframe Hand Detailer': nodes.BMABSubframeHandDetailer,
    'BMAB Resize By Person': nodes.BMABResizeByPerson,
    'BMAB Control Net': nodes.BMABControlNet,
    'BMAB Save Image': nodes.BMABSaveImage,
    'BMAB Upscale With Model': nodes.BMABUpscaleWithModel,
    'BMAB LoRA Loader': nodes.BMABLoraLoader,
    'BMAB Prompt': nodes.BMABPrompt,
    'BMAB Resize and Fill': nodes.BMABResizeAndFill,
    'BMAB Google Gemini Prompt': nodes.BMABGoogleGemini,

    # Imaging
    'BMAB Detection Crop': nodes.BMABDetectionCrop,
    'BMAB Remove Background': nodes.BMABRemoveBackground,
    'BMAB Alpha Composit': nodes.BMABAlphaComposit,
    'BMAB Blend': nodes.BMABBlend,
    'BMAB Detect And Mask': nodes.BMABDetectAndMask,
    'BMAB Lama Inpaint': nodes.BMABLamaInpaint,

    # SD-WebUI API
    'BMAB SD-WebUI API Server': nodes.BMABApiServer,
    'BMAB SD-WebUI API T2I': nodes.BMABApiSDWebUIT2I,
    'BMAB SD-WebUI API T2I Hires.Fix': nodes.BMABApiSDWebUIT2IHiresFix,
    'BMAB SD-WebUI API BMAB Extension': nodes.BMABApiSDWebUIBMABExtension,
    'BMAB SD-WebUI API ControlNet': nodes.BMABApiSDWebUIControlNet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'Basic': 'BMAB Basic',
    'Edge': 'BMAB Edge',
    'DinoSam': 'BMAB DinoSam',
    'Integrator': 'BMAB Integrator',
    'BMAB KSampler': 'BMAB KSampler',
    'BMAB KSamplerHiresFix': 'BMAB KSampler Hires. Fix',
    'BMAB Upscaler': 'BMAB Upscaler',
    'BMAB Face Detailer': 'BMAB Face Detailer',
    'BMAB Person Detailer': 'BMAB Person Detailer',
    'BMAB Simple Hand Detailer': 'BMAB Simple Hand Detailer',
    'BMAB Subframe Hand Detailer': 'BMAB Subframe Hand Detailer',
    'BMAB Resize By Person': 'BMAB Resize By Person',
    'Extractor': 'BMAB Extractor',
    'SeedGenerator': 'BMAB Seed Generator',
    'BMAB Control Net': 'BMAB ControlNet',
    'BMAB Save Image': 'BMAB Save Image',
    'BMAB Upscale With Model': 'BMAB Upscale With Model',
    'BMAB LoRA Loader': 'BMAB Lora Loader',
    'BMAB Prompt': 'BMAB Prompt',
    'BMAB Resize and Fill': 'BMAB Resize And Fill',
    'BMAB Google Gemini Prompt': 'BMAB Google Gemini API',

    # Imaging
    'BMAB Detection Crop': 'BMAB Detection Crop',
    'BMAB Remove Background': 'BMAB Remove Background',
    'BMAB Alpha Composit': 'BMAB Alpha Composit',
    'BMAB Blend': 'BMAB Blend',
    'BMAB Detect And Mask': 'BMAB Detect And Mask',
    'BMAB Lama Inpaint': 'BMAB Lama Inpaint',

    # SD-WebUI API
    'BMAB SD-WebUI API Server': 'BMAB SD-WebUI API Server',
    'BMAB SD-WebUI API T2I': 'BMAB SD-WebUI API T2I',
    'BMAB SD-WebUI API T2I Hires.Fix': 'BMAB SD-WebUI API T2I Hires.Fix',
    'BMAB SD-WebUI API BMAB Extension': 'BMAB SD-WebUI API BMAB Extension',
    'BMAB SD-WebUI API ControlNet': 'BMAB SD-WebUI API ControlNet',
}

