from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from .models import ContactMessage

class RegisterForm(forms.Form):
    full_name = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your full name',
            'required': 'required'
        }),
        label="Full Name"
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address',
            'required': 'required'
        }),
        label="Email Address"
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter password',
            'required': 'required'
        }),
        label="Password"
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirm password',
            'required': 'required'
        }),
        label="Confirm Password"
    )

    def clean_email(self):
        email = self.cleaned_data.get('email', '').strip().lower()
        if User.objects.filter(username=email).exists() or User.objects.filter(email=email).exists():
            raise ValidationError("An account with this email address already exists.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')

        if password and confirm_password and password != confirm_password:
            self.add_error('confirm_password', "Passwords do not match.")
        
        if password and len(password) < 6:
            self.add_error('password', "Password must be at least 6 characters long.")
            
        return cleaned_data

    def save(self):
        email = self.cleaned_data.get('email')
        full_name = self.cleaned_data.get('full_name')
        password = self.cleaned_data.get('password')

        # Split full name into first and last name
        name_parts = full_name.strip().split(' ', 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ''

        user = User.objects.create_user(
            username=email,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name
        )
        return user


class LoginForm(forms.Form):
    email = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email or username',
            'required': 'required'
        }),
        label="Email Address / Username"
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your password',
            'required': 'required'
        }),
        label="Password"
    )


class ContactForm(forms.ModelForm):
    class Meta:
        model = ContactMessage
        fields = ['name', 'email', 'message']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Your Full Name', 'required': 'required'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Your Email Address', 'required': 'required'}),
            'message': forms.Textarea(attrs={'class': 'form-control', 'placeholder': 'Write your message here...', 'rows': 5, 'required': 'required'}),
        }
