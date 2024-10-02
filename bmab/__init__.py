from bmab import nodes, serverext


NODE_CLASS_MAPPINGS = {
    # Basic
    'BMAB Basic': nodes.BMABBasic,
    'BMAB Edge': nodes.BMABEdge,
    'BMAB Text': nodes.BMABText,
    'BMAB Preview Text': nodes.BMABPreviewText,

    # Resize
    'BMAB Resize By Person': nodes.BMABResizeByPerson,
    'BMAB Resize By Ratio': nodes.BMABResizeByRatio,
    'BMAB Resize and Fill': nodes.BMABResizeAndFill,
    'BMAB Crop': nodes.BMABCrop,
    'BMAB Zoom Out': nodes.BMABZoomOut,
    'BMAB Square': nodes.BMABSquare,

    # Sampler
    'BMAB Integrator': nodes.BMABIntegrator,
    'BMAB ToBind': nodes.BMABToBind,
    'BMAB Flux Integrator': nodes.BMABFluxIntegrator,
    'BMAB Extractor': nodes.BMABExtractor,
    'BMAB SeedGenerator': nodes.BMABSeedGenerator,
    'BMAB KSampler': nodes.BMABKSampler,
    'BMAB KSamplerHiresFix': nodes.BMABKSamplerHiresFix,
    'BMAB KSamplerHiresFixWithUpscaler': nodes.BMABKSamplerHiresFixWithUpscaler,
    'BMAB Context': nodes.BMABContextNode,
    'BMAB Import Integrator': nodes.BMABImportIntegrator,
    'BMAB KSamplerKohyaDeepShrink': nodes.BMABKSamplerKohyaDeepShrink,
    'BMAB Clip Text Encoder SDXL': nodes.BMABClipTextEncoderSDXL,

    # Detailer
    'BMAB Face Detailer': nodes.BMABFaceDetailer,
    'BMAB Person Detailer': nodes.BMABPersonDetailer,
    'BMAB Simple Hand Detailer': nodes.BMABSimpleHandDetailer,
    'BMAB Subframe Hand Detailer': nodes.BMABSubframeHandDetailer,
    'BMAB Openpose Hand Detailer': nodes.BMABOpenposeHandDetailer,
    'BMAB Detail Anything': nodes.BMABDetailAnything,

    # Control Net
    'BMAB ControlNet': nodes.BMABControlNet,
    'BMAB Flux ControlNet': nodes.BMABFluxControlNet,
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
    'BMAB Load Output Image': nodes.BMABLoadOutputImage,
    'BMAB Black And White': nodes.BMABBlackAndWhite,
    'BMAB Detect And Paste': nodes.BMABDetectAndPaste,

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
    'BMAB Noise Generator': nodes.BMABNoiseGenerator,

    # Watermark
    'BMAB Watermark': nodes.BMABWatermark,

    'BMAB Upscaler': nodes.BMABUpscale,
    'BMAB Save Image': nodes.BMABSaveImage,
    'BMAB Remote Access And Save': nodes.BMABRemoteAccessAndSave,
    'BMAB Upscale With Model': nodes.BMABUpscaleWithModel,
    'BMAB LoRA Loader': nodes.BMABLoraLoader,
    'BMAB Prompt': nodes.BMABPrompt,
    'BMAB Google Gemini Prompt': nodes.BMABGoogleGemini,

    # Fill
    'BMAB Reframe': nodes.BMABReframe,
    'BMAB Outpaint By Ratio': nodes.BMABOutpaintByRatio,
    'BMAB Inpaint': nodes.BMABInpaint
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Preview
    'BMAB Basic': 'BMAB Basic',
    'BMAB Edge': 'BMAB Edge',
    'BMAB Text': 'BMAB Text',
    'BMAB Preview Text': 'BMAB Preview Text',

    # Resize
    'BMAB Resize By Person': 'BMAB Resize By Person',
    'BMAB Resize By Ratio': 'BMAB Resize By Ratio',
    'BMAB Resize and Fill': 'BMAB Resize And Fill',
    'BMAB Crop': 'BMAB Crop',
    'BMAB Zoom Out': 'BMAB Zoom Out',
    'BMAB Square': 'BMAB Square',

    # Sampler
    'BMAB Integrator': 'BMAB Integrator',
    'BMAB ToBind': 'BMAB ToBind',
    'BMAB Flux Integrator': 'BMAB Flux Integrator',
    'BMAB KSampler': 'BMAB KSampler',
    'BMAB KSamplerHiresFix': 'BMAB KSampler Hires. Fix',
    'BMAB KSamplerHiresFixWithUpscaler': 'BMAB KSampler Hires. Fix With Upscaler',
    'BMAB Extractor': 'BMAB Extractor',
    'BMAB SeedGenerator': 'BMAB Seed Generator',
    'BMAB Context': 'BMAB Context',
    'BMAB Import Integrator': 'BMAB Import Integrator',
    'BMAB KSamplerKohyaDeepShrink': 'BMAB KSampler with Kohya Deep Shrink',
    'BMAB Clip Text Encoder SDXL': 'BMAB Clip Text Encoder SDXL',

    # Detailer
    'BMAB Face Detailer': 'BMAB Face Detailer',
    'BMAB Person Detailer': 'BMAB Person Detailer',
    'BMAB Simple Hand Detailer': 'BMAB Simple Hand Detailer',
    'BMAB Subframe Hand Detailer': 'BMAB Subframe Hand Detailer',
    'BMAB Openpose Hand Detailer': 'BMAB Openpose Hand Detailer',
    'BMAB Detail Anything': 'BMAB Detail Anything',

    # Control Net
    'BMAB ControlNet': 'BMAB ControlNet',
    'BMAB Flux ControlNet': 'BMAB Flux ControlNet',
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
    'BMAB Load Output Image': 'BMAB Load Output Image',
    'BMAB Black And White': 'BMAB Black And White',
    'BMAB Detect And Paste': 'BMAB Detect And Paste',

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
    'BMAB Noise Generator': 'BMAB Noise Generator',

    # Watermark
    'BMAB Watermark': 'BMAB Watermark',

    'BMAB DinoSam': 'BMAB DinoSam',
    'BMAB Upscaler': 'BMAB Upscaler',
    'BMAB Control Net': 'BMAB ControlNet',
    'BMAB Save Image': 'BMAB Save Image',
    'BMAB Remote Access And Save': 'BMAB Remote Access And Save',
    'BMAB Upscale With Model': 'BMAB Upscale With Model',
    'BMAB LoRA Loader': 'BMAB Lora Loader',
    'BMAB Prompt': 'BMAB Prompt',
    'BMAB Google Gemini Prompt': 'BMAB Google Gemini API',

    # Fill
    'BMAB Reframe': 'BMAB Reframe',
    'BMAB Outpaint By Ratio': 'BMAB Outpaint By Ratio',
    'BMAB Inpaint': 'BMAB Inpaint',
}

