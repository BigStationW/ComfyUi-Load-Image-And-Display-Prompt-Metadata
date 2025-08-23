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

// More generalist function to find positive/negative prompts in the workflow
function extractPromptsFromWorkflow(workflow) {
    const prompts = { positive: "", negative: "" };
    if (!workflow) return prompts;

    // Configuration for input patterns
    const INPUT_PATTERNS = {
        positive: ['positive', 'conditioning_positive', 'pos'],
        negative: ['negative', 'conditioning_negative', 'nag_negative', 'neg']
    };

    try {
        // Step 1: Find all nodes that have positive/negative inputs
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
                        node,
                        hasPositive,
                        hasNegative,
                        priority: (hasPositive ? 1 : 0) + (hasNegative ? 1 : 0)
                    });
                }
            }
        }

        // Step 2: Sort by priority (nodes with both positive and negative first)
        candidateNodes.sort((a, b) => b.priority - a.priority);

        // Step 3: Extract prompts from the best candidates
        let positiveNodeId = null;
        let negativeNodeId = null;

        for (const candidate of candidateNodes) {
            const node = candidate.node;
            
            // Check for positive prompt connection
            if (!positiveNodeId) {
                for (const pattern of INPUT_PATTERNS.positive) {
                    if (node.inputs[pattern] && Array.isArray(node.inputs[pattern])) {
                        positiveNodeId = String(node.inputs[pattern][0]);
                        break;
                    }
                }
            }

            // Check for negative prompt connection
            if (!negativeNodeId) {
                for (const pattern of INPUT_PATTERNS.negative) {
                    if (node.inputs[pattern] && Array.isArray(node.inputs[pattern])) {
                        negativeNodeId = String(node.inputs[pattern][0]);
                        break;
                    }
                }
            }

            // If we found both, we can break
            if (positiveNodeId && negativeNodeId) break;
        }

        // Step 4: If still no connections found, look for any input that might be conditioning
        if (!positiveNodeId || !negativeNodeId) {
            for (const nodeId in workflow) {
                const node = workflow[nodeId];
                if (node.inputs) {
                    // Look for any input that connects to text encoding nodes
                    for (const inputKey in node.inputs) {
                        const inputValue = node.inputs[inputKey];
                        if (Array.isArray(inputValue) && inputValue.length > 0) {
                            const connectedNode = workflow[String(inputValue[0])];
                            if (connectedNode && isTextEncodingNode(connectedNode)) {
                                // Use naming patterns to determine positive vs negative
                                const isNegative = inputKey.toLowerCase().includes('negative') || 
                                                 inputKey.toLowerCase().includes('nag') ||
                                                 node._meta?.title?.toLowerCase().includes('negative');
                                
                                if (isNegative && !negativeNodeId) {
                                    negativeNodeId = String(inputValue[0]);
                                } else if (!isNegative && !positiveNodeId) {
                                    positiveNodeId = String(inputValue[0]);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Helper function to check if a node is a text encoding node
        function isTextEncodingNode(node) {
            const textEncodingTypes = [
                'CLIPTextEncode',
                'CLIPTextEncodeFlux',
                'StringConcatenate',
                'String Literal',
                'ImpactConcatConditionings',
                'PCLazyTextEncode',
                'ImpactWildcardProcessor',
                'TextEncodeQwenImageEdit'
            ];
            return textEncodingTypes.includes(node.class_type);
        }

        // Helper function to extract text from a node, following concatenations
        function extractTextFromNode(nodeId, visited = new Set()) {
            if (!nodeId || visited.has(nodeId)) return "";
            visited.add(nodeId);
            
            const node = workflow[String(nodeId)];
            if (!node) return "";
            
            // Handle ImpactWildcardProcessor node
            if (node.class_type === "ImpactWildcardProcessor" && node.inputs) {
                // Use populated_text if available, otherwise use wildcard_text
                return node.inputs.populated_text || node.inputs.wildcard_text || "";
            }
            
            // Handle PCLazyTextEncode node
            else if (node.class_type === "PCLazyTextEncode" && node.inputs && node.inputs.text) {
                // Check if text is a reference to another node
                if (Array.isArray(node.inputs.text)) {
                    return extractTextFromNode(String(node.inputs.text[0]), visited);
                }
                // Otherwise it's the actual text
                return node.inputs.text;
            }
            
            // Handle ChromaPaddingRemoval - pass through to its input
            else if (node.class_type === "ChromaPaddingRemoval" && node.inputs && node.inputs.conditioning) {
                if (Array.isArray(node.inputs.conditioning)) {
                    return extractTextFromNode(String(node.inputs.conditioning[0]), visited);
                }
            }
            
            // Handle ImpactConcatConditionings node
            else if (node.class_type === "ImpactConcatConditionings" && node.inputs) {
                let concatenatedText = [];
                
                // Check all conditioning inputs (conditioning1, conditioning2, etc.)
                for (let i = 1; i <= 10; i++) {
                    const condInput = node.inputs[`conditioning${i}`];
                    if (condInput && Array.isArray(condInput)) {
                        const text = extractTextFromNode(String(condInput[0]), visited);
                        if (text) concatenatedText.push(text);
                    }
                }
                
                return concatenatedText.join("\n\n");
            }
            
            // Handle CLIPTextEncode nodes
            else if (node.class_type === "CLIPTextEncode" && node.inputs && node.inputs.text) {
                // Check if text is a reference to another node
                if (Array.isArray(node.inputs.text)) {
                    return extractTextFromNode(String(node.inputs.text[0]), visited);
                }
                // Otherwise it's the actual text
                return node.inputs.text;
            }
            
            // Handle CLIPTextEncodeFlux nodes
            else if (node.class_type === "CLIPTextEncodeFlux" && node.inputs) {
                const text = node.inputs.clip_l || node.inputs.t5xxl || "";
                // Check if it's a reference
                if (Array.isArray(text)) {
                    return extractTextFromNode(String(text[0]), visited);
                }
                return text;
            }
            
            // Handle StringConcatenate nodes
            else if (node.class_type === "StringConcatenate" && node.inputs) {
                let parts = [];
                
                // Get string_a
                if (node.inputs.string_a) {
                    if (Array.isArray(node.inputs.string_a)) {
                        parts.push(extractTextFromNode(String(node.inputs.string_a[0]), visited));
                    } else {
                        parts.push(node.inputs.string_a);
                    }
                }
                
                // Get delimiter
                const delimiter = node.inputs.delimiter || "";
                
                // Get string_b
                if (node.inputs.string_b) {
                    if (Array.isArray(node.inputs.string_b)) {
                        parts.push(extractTextFromNode(String(node.inputs.string_b[0]), visited));
                    } else {
                        parts.push(node.inputs.string_b);
                    }
                }
                
                return parts.join(delimiter);
            }
            
            // Handle String Literal nodes
            else if (node.class_type === "String Literal" && node.inputs && node.inputs.string !== undefined) {
                return node.inputs.string;
            }

            // Handle TextEncodeQwenImageEdit nodes
            else if (node.class_type === "TextEncodeQwenImageEdit" && node.inputs && node.inputs.prompt) {
                // Check if prompt is a reference to another node
                if (Array.isArray(node.inputs.prompt)) {
                    return extractTextFromNode(String(node.inputs.prompt[0]), visited);
                }
                // Otherwise it's the actual text
                return node.inputs.prompt;
            }
            
            return "";
        }

        // Extract prompts using the found node IDs
        if (positiveNodeId) {
            prompts.positive = extractTextFromNode(positiveNodeId);
        }

        if (negativeNodeId) {
            prompts.negative = extractTextFromNode(negativeNodeId);
        }

        // Fallback: if we still didn't find prompts, check by node titles and types
        if (!prompts.positive || !prompts.negative) {
            for (const nodeId in workflow) {
                const node = workflow[nodeId];
                
                if (node.class_type === "CLIPTextEncode" && node.inputs && node.inputs.text) {
                    const title = node._meta?.title?.toLowerCase() || "";
                    
                    if (title.includes("negative") || title.includes("nag")) {
                        if (!prompts.negative) prompts.negative = node.inputs.text;
                    } else {
                        if (!prompts.positive && nodeId !== positiveNodeId && nodeId !== negativeNodeId) {
                            prompts.positive = node.inputs.text;
                        }
                    }
                }
                
                // Handle CLIPTextEncodeFlux nodes in fallback
                else if (node.class_type === "CLIPTextEncodeFlux" && node.inputs) {
                    const text = node.inputs.clip_l || node.inputs.t5xxl || "";
                    const title = node._meta?.title?.toLowerCase() || "";
                    
                    if (title.includes("negative") || title.includes("nag")) {
                        if (!prompts.negative) prompts.negative = text;
                    } else {
                        if (!prompts.positive && nodeId !== positiveNodeId && nodeId !== negativeNodeId) {
                            prompts.positive = text;
                        }
                    }
                }
            }
        }
        
    } catch (error) {
        logError("[LoadImageX] Error extracting prompts from workflow:", error);
    }
    
    return prompts;
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
