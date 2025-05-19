# Zsh and Oh My Zsh
sudo apt install zsh -y
RUNZSH=no CHSH=no KEEP_ZSHRC=yes sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
chsh -s "$(which zsh)"

# Plugins
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

sudo apt install bat -y
git clone https://github.com/fdellwing/zsh-bat.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-bat

# edit plugins=() in ~/.zshrc
plugins=(git zsh-autosuggestions zsh-syntax-highlighting zsh-bat)
sed -i 's/^plugins=(.*)$/plugins=(git zsh-autosuggestions zsh-syntax-highlighting zsh-bat)/' ~/.zshrc

# direnv
curl -sfL https://direnv.net/install.sh | bash

if ! grep -Fxq 'eval "$(direnv hook zsh)"' ~/.zshrc; then
    echo >> ~/.zshrc
    echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
fi

# starship rs
curl -sS https://starship.rs/install.sh | sh -s -- -y

if ! grep -Fxq 'eval "$(starship init zsh)"' ~/.zshrc; then
    echo >> ~/.zshrc
    echo 'eval "$(starship init zsh)"' >> ~/.zshrc
fi


omz reload