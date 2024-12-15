#! bash
echo "Downloading https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip "
echo '$ curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"'
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

echo 'importing GPG key ./aws_cli.pgp'
echo '$ gpg --import aws_cli.pgp'
gpg --import aws_cli.pgp

echo "Downloading signature file []https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip.sig] "
echo '$ curl -o awscliv2.sig https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip.sig'
curl -o awscliv2.sig https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip.sig


echo "Verifying signature"
echo '$ gpg --verify awscliv2.sig awscliv2.zip'
gpg --verify awscliv2.sig awscliv2.zip

echo "unziping file"
echo '$ unzip -qu awscliv2.zip'
unzip -qu awscliv2.zip

echo "installing AWS cli"
# sudo ./aws/install
echo '$ ./aws/install -i ./awscli -b ./awscli'
./aws/install -i $PWD/awscli -b $PWD/awscli

echo "adjusting local PATH"
echo '$ export PATH=./awscli:$PATH'
export PATH=$PWD/awscli:$PATH
