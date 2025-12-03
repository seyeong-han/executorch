#!/bin/bash
set -e

SDK_VERSION="1.4.321.0"
SDK_FILENAME="vulkansdk-linux-x86_64-${SDK_VERSION}.tar.xz"
SDK_URL="https://sdk.lunarg.com/sdk/download/${SDK_VERSION}/linux/${SDK_FILENAME}"
INSTALL_DIR="/home/pe/vulkan-sdk"

echo "Downloading Vulkan SDK ${SDK_VERSION}..."
wget -q "${SDK_URL}" -O "${SDK_FILENAME}"

echo "Creating install directory..."
mkdir -p "${INSTALL_DIR}"

echo "Extracting SDK..."
tar -xf "${SDK_FILENAME}" -C "${INSTALL_DIR}"

echo "Cleaning up..."
rm "${SDK_FILENAME}"

echo "Vulkan SDK installed to ${INSTALL_DIR}/${SDK_VERSION}"
echo "Please source the setup script: source ${INSTALL_DIR}/${SDK_VERSION}/setup-env.sh"
