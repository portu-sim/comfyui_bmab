from bmab import nodes
from bmab import override


NODE_CLASS_MAPPINGS = {
    'BMAB Basic': nodes.BMABBasic,
    'BMAB Edge': nodes.BMABEdge,
    'BMAB Upscaler': nodes.BMABUpscale,
    'BMAB Resize By Person': nodes.BMABResizeByPerson,
    'BMAB Save Image': nodes.BMABSaveImage,
    'BMAB Upscale With Model': nodes.BMABUpscaleWithModel,
    'BMAB LoRA Loader': nodes.BMABLoraLoader,
    'BMAB Prompt': nodes.BMABPrompt,
    'BMAB Resize and Fill': nodes.BMABResizeAndFill,
    'BMAB Google Gemini Prompt': nodes.BMABGoogleGemini,

    # Sampler
    'BMAB Integrator': nodes.BMABIntegrator,
    'BMAB Extractor': nodes.BMABExtractor,
    'BMAB SeedGenerator': nodes.BMABSeedGenerator,
    'BMAB KSampler': nodes.BMABKSampler,
    'BMAB KSamplerHiresFix': nodes.BMABKSamplerHiresFix,
    'BMAB KSamplerHiresFixWithUpscaler': nodes.BMABKSamplerHiresFixWithUpscaler,
    'BMAB Context': nodes.BMABContextNode,

    # Detailer
    'BMAB Face Detailer': nodes.BMABFaceDetailer,
    'BMAB Person Detailer': nodes.BMABPersonDetailer,
    'BMAB Simple Hand Detailer': nodes.BMABSimpleHandDetailer,
    'BMAB Subframe Hand Detailer': nodes.BMABSubframeHandDetailer,
    'BMAB Openpose Hand Detailer': nodes.BMABOpenposeHandDetailer,
    'BMAB Detail Anything': nodes.BMABDetailAnything,

    # Control Net
    'BMAB ControlNet': nodes.BMABControlNet,
    'BMAB ControlNet Openpose': nodes.BMABControlNetOpenpose,
    'BMAB ControlNet IPAdapter': nodes.BMABControlNetIPAdapter,

    # Imaging
    'BMAB Detection Crop': nodes.BMABDetectionCrop,
    'BMAB Remove Background': nodes.BMABRemoveBackground,
    'BMAB Alpha Composit': nodes.BMABAlphaComposit,
    'BMAB Blend': nodes.BMABBlend,
    'BMAB Detect And Mask': nodes.BMABDetectAndMask,
    'BMAB Lama Inpaint': nodes.BMABLamaInpaint,
    'BMAB Detector': nodes.BMABDetector,
    'BMAB Segment Anything': nodes.BMABSegmentAnything,
    'BMAB Masks To Images': nodes.BMABMasksToImages,
    'BMAB Load Image': nodes.BMABLoadImage,

    # SD-WebUI API
    'BMAB SD-WebUI API Server': nodes.BMABApiServer,
    'BMAB SD-WebUI API T2I': nodes.BMABApiSDWebUIT2I,
    'BMAB SD-WebUI API I2I': nodes.BMABApiSDWebUII2I,
    'BMAB SD-WebUI API T2I Hires.Fix': nodes.BMABApiSDWebUIT2IHiresFix,
    'BMAB SD-WebUI API BMAB Extension': nodes.BMABApiSDWebUIBMABExtension,
    'BMAB SD-WebUI API ControlNet': nodes.BMABApiSDWebUIControlNet,

    # UTIL Nodes
    'BMAB Model To Bind': nodes.BMABModelToBind,
    'BMAB Conditioning To Bind': nodes.BMABConditioningToBind,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'BMAB Basic': 'BMAB Basic',
    'BMAB Edge': 'BMAB Edge',
    'BMAB DinoSam': 'BMAB DinoSam',
    'BMAB Upscaler': 'BMAB Upscaler',
    'BMAB Resize By Person': 'BMAB Resize By Person',
    'BMAB Control Net': 'BMAB ControlNet',
    'BMAB Save Image': 'BMAB Save Image',
    'BMAB Upscale With Model': 'BMAB Upscale With Model',
    'BMAB LoRA Loader': 'BMAB Lora Loader',
    'BMAB Prompt': 'BMAB Prompt',
    'BMAB Resize and Fill': 'BMAB Resize And Fill',
    'BMAB Google Gemini Prompt': 'BMAB Google Gemini API',

    # Sampler
    'BMAB Integrator': 'BMAB Integrator',
    'BMAB KSampler': 'BMAB KSampler',
    'BMAB KSamplerHiresFix': 'BMAB KSampler Hires. Fix',
    'BMAB KSamplerHiresFixWithUpscaler': 'BMAB KSampler Hires. Fix With Upscaler',
    'BMAB Extractor': 'BMAB Extractor',
    'BMAB SeedGenerator': 'BMAB Seed Generator',
    'BMAB Context': 'BMAB Context',

    # Detailer
    'BMAB Face Detailer': 'BMAB Face Detailer',
    'BMAB Person Detailer': 'BMAB Person Detailer',
    'BMAB Simple Hand Detailer': 'BMAB Simple Hand Detailer',
    'BMAB Subframe Hand Detailer': 'BMAB Subframe Hand Detailer',
    'BMAB Openpose Hand Detailer': 'BMAB Openpose Hand Detailer',
    'BMAB Detail Anything': 'BMAB Detail Anything',

    # Control Net
    'BMAB ControlNet': 'BMAB ControlNet',
    'BMAB ControlNet Openpose': 'BMAB ControlNet Openpose',
    'BMAB ControlNet IPAdapter': 'BMAB ControlNet IPAdapter',

    # Imaging
    'BMAB Detection Crop': 'BMAB Detection Crop',
    'BMAB Remove Background': 'BMAB Remove Background',
    'BMAB Alpha Composit': 'BMAB Alpha Composit',
    'BMAB Blend': 'BMAB Blend',
    'BMAB Detect And Mask': 'BMAB Detect And Mask',
    'BMAB Lama Inpaint': 'BMAB Lama Inpaint',
    'BMAB Detector': 'BMAB Detector',
    'BMAB Segment Anything': 'BMAB Segment Anything',
    'BMAB Masks To Images': 'BMAB Masks To Images',
    'BMAB Load Image': 'BMAB Load Image',

    # SD-WebUI API
    'BMAB SD-WebUI API Server': 'BMAB SD-WebUI API Server',
    'BMAB SD-WebUI API T2I': 'BMAB SD-WebUI API T2I',
    'BMAB SD-WebUI API I2I': 'BMAB SD-WebUI API I2I',
    'BMAB SD-WebUI API T2I Hires.Fix': 'BMAB SD-WebUI API T2I Hires.Fix',
    'BMAB SD-WebUI API BMAB Extension': 'BMAB SD-WebUI API BMAB Extension',
    'BMAB SD-WebUI API ControlNet': 'BMAB SD-WebUI API ControlNet',

    # UTIL Nodes
    'BMAB Model To Bind': 'BMAB Model To Bind',
    'BMAB Conditioning To Bind': 'BMAB Conditioning To Bind',
}

