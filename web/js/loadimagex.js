import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Configuration option to enable/disable logging
const ENABLE_LOGGING = false; // Set to true to enable all console logs

// Helper function for conditional logging
function log(message, style = "") {
    if (ENABLE_LOGGING) {
        if (style) {
            console.log(message, style);
        } else {
            console.log(message);
        }
    }
}

function logError(message, error) {
    if (ENABLE_LOGGING) {
        console.error(message, error);
    }
}

// Helper function to clean potential non-standard JSON from metadata
function cleanJSONString(jsonString) {
    if (!jsonString) return null;
    return jsonString.replace(/:\s*NaN/g, ': null');
}

// Self-contained function to parse metadata from a PNG file's raw data.
function parsePNGMetadata(arrayBuffer) {
    const dataView = new DataView(arrayBuffer);
    const metadata = {};

    // Check for PNG signature
    const pngSignature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    for (let i = 0; i < 8; i++) {
        if (dataView.getUint8(i) !== pngSignature[i]) {
            logError("[LoadImageX] Not a valid PNG file.");
            return null;
        }
    }

    let offset = 8;
    while (offset < dataView.byteLength) {
        const length = dataView.getUint32(offset);
        offset += 4;
        const chunkType = String.fromCharCode(
            dataView.getUint8(offset), dataView.getUint8(offset + 1),
            dataView.getUint8(offset + 2), dataView.getUint8(offset + 3)
        );
        offset += 4;

        if (chunkType === 'tEXt' || chunkType === 'iTXt') {
            const chunkData = new Uint8Array(arrayBuffer, offset, length);
            const text = new TextDecoder('utf-8').decode(chunkData);
            const nullIndex = text.indexOf('\0');
            if (nullIndex !== -1) {
                const keyword = text.substring(0, nullIndex);
                const value = text.substring(nullIndex + 1);
                metadata[keyword] = value;
            }
        }

        offset += length + 4; // Skip data and CRC
        if (chunkType === 'IEND') break;
    }
    return metadata;
}

// Robust function to find positive/negative prompts in the workflow
function extractPromptsFromWorkflow(workflow) {
    const prompts = { positive: "", negative: "" };
    if (!workflow) return prompts;

    // 1. PRIORITY STRATEGY: Look for nodes explicitly named "Positive Prompt" or "Negative Prompt"
    // This is the most reliable method for templates and organized workflows.
    for (const nodeId in workflow) {
        const node = workflow[nodeId];
        const title = node._meta?.title?.toLowerCase() || "";
        
        if (title === "positive prompt" || title === "pos prompt" || title === "prompt") {
            // Only overwrite if we haven't found a text yet or if this seems more "direct"
            const text = extractTextFromNode(nodeId, new Set(), workflow);
            if (text) prompts.positive = text;
        }
        
        if (title === "negative prompt" || title === "neg prompt") {
            const text = extractTextFromNode(nodeId, new Set(), workflow);
            if (text) prompts.negative = text;
        }
    }

    // If we found both via titles, we are done.
    if (prompts.positive && prompts.negative) return prompts;

    // 2. GRAPH TRAVERSAL STRATEGY: Follow the connections from Samplers
    const INPUT_PATTERNS = {
        // Added 'guider' for SamplerCustomAdvanced/Flux workflows
        positive: ['positive', 'conditioning_positive', 'pos', 'guider'], 
        negative: ['nag_negative', 'negative', 'conditioning_negative', 'neg']
    };

    try {
        // Find candidate nodes (nodes with positive/negative/guider inputs)
        const candidateNodes = [];
        for (const nodeId in workflow) {
            const node = workflow[nodeId];
            if (node.inputs) {
                const hasPositive = INPUT_PATTERNS.positive.some(pattern => 
                    node.inputs[pattern] && Array.isArray(node.inputs[pattern])
                );
                const hasNegative = INPUT_PATTERNS.negative.some(pattern => 
                    node.inputs[pattern] && Array.isArray(node.inputs[pattern])
                );
                
                if (hasPositive || hasNegative) {
                    candidateNodes.push({
                        nodeId,
                        hasPositive,
                        hasNegative,
                        priority: (hasPositive ? 1 : 0) + (hasNegative ? 1 : 0)
                    });
                }
            }
        }
        candidateNodes.sort((a, b) => b.priority - a.priority);

        let positiveNodeId = null;
        let negativeNodeId = null;

        for (const candidate of candidateNodes) {
            const node = workflow[candidate.nodeId];
            if (!positiveNodeId && !prompts.positive) {
                for (const pattern of INPUT_PATTERNS.positive) {
                    if (node.inputs[pattern] && Array.isArray(node.inputs[pattern])) {
                        positiveNodeId = String(node.inputs[pattern][0]);
                        break;
                    }
                }
            }
            if (!negativeNodeId && !prompts.negative) {
                for (const pattern of INPUT_PATTERNS.negative) {
                    if (node.inputs[pattern] && Array.isArray(node.inputs[pattern])) {
                        negativeNodeId = String(node.inputs[pattern][0]);
                        break;
                    }
                }
            }
            if ((positiveNodeId || prompts.positive) && (negativeNodeId || prompts.negative)) break;
        }

        // Extract using the IDs found via graph traversal (if not already found by title)
        if (positiveNodeId && !prompts.positive) prompts.positive = extractTextFromNode(positiveNodeId, new Set(), workflow);
        if (negativeNodeId && !prompts.negative) prompts.negative = extractTextFromNode(negativeNodeId, new Set(), workflow);

        // 3. FALLBACK STRATEGY: Scan for unconnected CLIPTextEncode nodes
        if (!prompts.positive || !prompts.negative) {
            for (const nodeId in workflow) {
                const node = workflow[nodeId];
                if ((node.class_type === "CLIPTextEncode" || node.class_type === "CLIPTextEncodeFlux") && node.inputs && typeof node.inputs.text === 'string') {
                    const title = node._meta?.title?.toLowerCase() || "";
                    if (title.includes("negative") || title.includes("nag")) {
                        if (!prompts.negative) prompts.negative = node.inputs.text;
                    } else {
                        if (!prompts.positive && nodeId !== positiveNodeId && nodeId !== negativeNodeId) {
                            prompts.positive = node.inputs.text;
                        }
                    }
                }
            }
        }
        
    } catch (error) {
        logError("[LoadImageX] Error processing workflow:", error);
    }
    
    return prompts;
}

// Helper function extracted outside to be clean and recursive
function extractTextFromNode(nodeId, visited = new Set(), workflow) {
    if (!nodeId || visited.has(nodeId)) return "";
    visited.add(nodeId);
    
    const node = workflow[String(nodeId)];
    if (!node) return "";

    // --- TRAVERSAL NODES ---

    // 1. BasicGuider (Flux/SD3)
    // Pass through to the 'conditioning' input
    if (node.class_type === "BasicGuider") {
        if (node.inputs && node.inputs.conditioning && Array.isArray(node.inputs.conditioning)) {
            return extractTextFromNode(String(node.inputs.conditioning[0]), visited, workflow);
        }
    }

    // 2. ConditioningCombine
    if (node.class_type === "ConditioningCombine") {
        const t1 = node.inputs.conditioning_1 ? extractTextFromNode(String(node.inputs.conditioning_1[0]), visited, workflow) : "";
        const t2 = node.inputs.conditioning_2 ? extractTextFromNode(String(node.inputs.conditioning_2[0]), visited, workflow) : "";
        return [t1, t2].filter(t => t).join("\n");
    }

    // 3. Conditioning Passthroughs
    const condPassthroughTypes = [
        "ConditioningSetTimestepRange", "ConditioningAverage", "ConditioningSetArea", 
        "ConditioningSetMask", "ChromaPaddingRemoval", "ConditioningZeroOut"
    ];
    if (condPassthroughTypes.includes(node.class_type)) {
        if (node.class_type === "ConditioningZeroOut") return ""; // Stop at ZeroOut usually
        
        if (node.inputs.conditioning && Array.isArray(node.inputs.conditioning)) {
            return extractTextFromNode(String(node.inputs.conditioning[0]), visited, workflow);
        }
        if (node.inputs.conditioning_to && Array.isArray(node.inputs.conditioning_to)) {
            return extractTextFromNode(String(node.inputs.conditioning_to[0]), visited, workflow);
        }
    }

    // --- TEXT EXTRACTION NODES ---

    // 4. Text Concatenate (WAS Node Suite & Others)
    if ((node.class_type === "Text Concatenate" || node.class_type === "StringConcatenate") && node.inputs) {
        let parts = [];
        const delimiter = node.inputs.delimiter || "";
        
        // WAS Suite style
        ['text_a', 'text_b', 'text_c', 'text_d'].forEach(key => {
            if (node.inputs[key]) {
                if (Array.isArray(node.inputs[key])) {
                    const val = extractTextFromNode(String(node.inputs[key][0]), visited, workflow);
                    if(val) parts.push(val);
                } else {
                    parts.push(node.inputs[key]);
                }
            }
        });

        // Custom Scripts style
        if (parts.length === 0) {
             if (node.inputs.string_a) {
                 if(Array.isArray(node.inputs.string_a)) parts.push(extractTextFromNode(String(node.inputs.string_a[0]), visited, workflow));
                 else parts.push(node.inputs.string_a);
             }
             if (node.inputs.string_b) {
                 if(Array.isArray(node.inputs.string_b)) parts.push(extractTextFromNode(String(node.inputs.string_b[0]), visited, workflow));
                 else parts.push(node.inputs.string_b);
             }
        }
        
        return parts.join(delimiter);
    }

    // 5. Text Multiline (WAS Node Suite)
    if (node.class_type === "Text Multiline" && node.inputs) {
        const textVal = node.inputs.text;
        if (Array.isArray(textVal)) return extractTextFromNode(String(textVal[0]), visited, workflow);
        return textVal || "";
    }

    // 6. CLIPTextEncode (Standard, Flux, PCLazy)
    // Added PCLazyTextEncodeAdvanced support specifically
    if ((node.class_type === "CLIPTextEncode" || node.class_type === "CLIPTextEncodeFlux" || 
         node.class_type === "PCLazyTextEncode" || node.class_type === "PCLazyTextEncodeAdvanced") && node.inputs) {
        const textKeys = ['text', 'clip_l', 't5xxl'];
        for (const key of textKeys) {
            if (node.inputs[key] !== undefined) {
                if (Array.isArray(node.inputs[key])) {
                    return extractTextFromNode(String(node.inputs[key][0]), visited, workflow);
                }
                const value = node.inputs[key] || "";
                if (value) return value;
            }
        }
    }

    // 7. ImpactWildcardProcessor
    if (node.class_type === "ImpactWildcardProcessor" && node.inputs) {
        return node.inputs.populated_text || node.inputs.wildcard_text || "";
    }

    // 8. String Literal
    if (node.class_type === "String Literal" && node.inputs) {
        return node.inputs.string || "";
    }

    // 9. Primitive Node (sometimes text is stored in a primitive node)
    if (node.class_type === "PrimitiveNode" && node.inputs) {
        // Check widgets_values if they exist (common in primitive nodes)
        if (node.widgets_values && node.widgets_values.length > 0) {
             const val = node.widgets_values[0];
             if (typeof val === 'string') return val;
        }
    }

    // 10. General Fallback
    if (node.inputs) {
         if (node.inputs.text) {
             if (Array.isArray(node.inputs.text)) {
                 return extractTextFromNode(String(node.inputs.text[0]), visited, workflow);
             } else if (typeof node.inputs.text === 'string') {
                 return node.inputs.text;
             }
         }
         // Try following conditioning if we haven't found a specific type
         if (node.inputs.conditioning && Array.isArray(node.inputs.conditioning)) {
             return extractTextFromNode(String(node.inputs.conditioning[0]), visited, workflow);
         }
    }

    return "";
}

// Main function to get metadata from an image and update the text boxes
async function updatePromptsFromImage(filename, node) {
    const positiveWidget = node.widgets.find(w => w.name === "positive_prompt");
    const negativeWidget = node.widgets.find(w => w.name === "negative_prompt");

    if (positiveWidget) positiveWidget.value = "";
    if (negativeWidget) negativeWidget.value = "";

    try {
        const res = await api.fetchApi(`/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=`);
        if (!res.ok) throw new Error(`Failed to fetch image: ${res.status}`);
        
        const buffer = await res.arrayBuffer();
        const metadata = parsePNGMetadata(buffer);
        
        // Log metadata in blue
        log("%c[LoadImageX] Metadata:", "color: #0066ff; font-weight: bold");
        log("%c" + JSON.stringify(metadata, null, 2), "color: #0066ff");
        
        if (metadata && metadata.prompt) {
            const promptData = JSON.parse(cleanJSONString(metadata.prompt));
            const prompts = extractPromptsFromWorkflow(promptData);

            // Log positive prompt in green
            if (prompts.positive) {
                log("%c[LoadImageX] Positive Prompt:", "color: #00cc00; font-weight: bold");
                log("%c" + prompts.positive, "color: #00cc00");
            }
            
            // Log negative prompt in red
            if (prompts.negative) {
                log("%c[LoadImageX] Negative Prompt:", "color: #ff0000; font-weight: bold");
                log("%c" + prompts.negative, "color: #ff0000");
            }

            if (positiveWidget && prompts.positive) positiveWidget.value = prompts.positive;
            if (negativeWidget && prompts.negative) negativeWidget.value = prompts.negative;
        }
    } catch (error) {
        logError("[LoadImageX] Error processing image metadata:", error);
    }
}

app.registerExtension({
    name: "testt.LoadImageX",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LoadImageX") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const self = this;
                
                const imageWidget = this.widgets.find(w => w.name === "image");
                if (!imageWidget) return r;

                // Store the original callback
                const originalCallback = imageWidget.callback;
                
                // Override the callback to handle both value change and preview update
                imageWidget.callback = function(value) {
                    if (value) {
                        // Update prompts from metadata
                        updatePromptsFromImage(value, self);
                        
                        // Trigger the node to update its outputs
                        // This is important for the preview to update
                        if (self.graph) {
                            self.graph.runStep();
                        }
                    }
                    
                    // Call original callback if it exists
                    if (originalCallback) {
                        return originalCallback.apply(this, arguments);
                    }
                };

                // If there's already a default image value, load its metadata
                if (imageWidget.value) {
                    // Use setTimeout to ensure the node is fully initialized
                    setTimeout(() => {
                        updatePromptsFromImage(imageWidget.value, self);
                    }, 100);
                }

                // Handle the upload button
                const uploadWidget = this.widgets.find(w => w.type === "button");
                if (uploadWidget) {
                    uploadWidget.callback = () => {
                        const fileInput = document.createElement("input");
                        fileInput.type = "file";
                        fileInput.accept = "image/png,image/jpeg,image/webp";
                        fileInput.style.display = "none";
                        document.body.appendChild(fileInput);

                        fileInput.onchange = async (e) => {
                            if (!e.target.files.length) {
                                document.body.removeChild(fileInput);
                                return;
                            }
                            const file = e.target.files[0];
                            const formData = new FormData();
                            formData.append("image", file);
                            formData.append("overwrite", "true");
                            
                            try {
                                const response = await api.fetchApi("/upload/image", { 
                                    method: "POST", 
                                    body: formData 
                                });
                                if (response.ok) {
                                    const data = await response.json();
                                    // Update the widget value
                                    imageWidget.value = data.name;
                                    // Trigger the callback to update prompts and preview
                                    if (imageWidget.callback) {
                                        imageWidget.callback(data.name);
                                    }
                                } else {
                                    logError("[LoadImageX] Upload failed:", response.statusText);
                                }
                            } catch (error) {
                                logError("[LoadImageX] Upload error:", error);
                            } finally {
                                document.body.removeChild(fileInput);
                            }
                        };
                        fileInput.click();
                    };
                }
                
                return r;
            };
        }
    }
});

// NEW: OnlyLoadImagesWithMetadata extension
app.registerExtension({
    name: "testt.OnlyLoadImagesWithMetadata",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "OnlyLoadImagesWithMetadata") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const self = this;
                
                const imageWidget = this.widgets.find(w => w.name === "image");
                if (!imageWidget) return r;

                // Store the original callback
                const originalCallback = imageWidget.callback;
                
                // Override the callback to handle both value change and preview update
                imageWidget.callback = function(value) {
                    if (value) {
                        // Update prompts from metadata
                        updatePromptsFromImage(value, self);
                        
                        // Trigger the node to update its outputs
                        // This is important for the preview to update
                        if (self.graph) {
                            self.graph.runStep();
                        }
                    }
                    
                    // Call original callback if it exists
                    if (originalCallback) {
                        return originalCallback.apply(this, arguments);
                    }
                };

                // If there's already a default image value, load its metadata
                if (imageWidget.value) {
                    // Use setTimeout to ensure the node is fully initialized
                    setTimeout(() => {
                        updatePromptsFromImage(imageWidget.value, self);
                    }, 100);
                }

                // Handle the upload button
                const uploadWidget = this.widgets.find(w => w.type === "button");
                if (uploadWidget) {
                    uploadWidget.callback = () => {
                        const fileInput = document.createElement("input");
                        fileInput.type = "file";
                        fileInput.accept = ".png";
                        fileInput.style.display = "none";
                        document.body.appendChild(fileInput);

                        fileInput.onchange = async (e) => {
                            if (!e.target.files.length) {
                                document.body.removeChild(fileInput);
                                return;
                            }
                            const file = e.target.files[0];
                            const formData = new FormData();
                            formData.append("image", file);
                            formData.append("overwrite", "true");
                            
                            try {
                                const response = await api.fetchApi("/upload/image", { 
                                    method: "POST", 
                                    body: formData 
                                });
                                if (response.ok) {
                                    const data = await response.json();
                                    // Update the widget value
                                    imageWidget.value = data.name;
                                    // Trigger the callback to update prompts and preview
                                    if (imageWidget.callback) {
                                        imageWidget.callback(data.name);
                                    }
                                } else {
                                    logError("[OnlyLoadImagesWithMetadata] Upload failed:", response.statusText);
                                }
                            } catch (error) {
                                logError("[OnlyLoadImagesWithMetadata] Upload error:", error);
                            } finally {
                                document.body.removeChild(fileInput);
                            }
                        };
                        fileInput.click();
                    };
                }
                
                return r;
            };
        }
    }
});
